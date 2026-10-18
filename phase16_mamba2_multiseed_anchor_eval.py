import argparse
import copy
import csv
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from output_naming import build_output_path

print("[phase16] importing torch...")
import torch
print("[phase16] importing transformers...")
from transformers import AutoTokenizer

print("[phase16] importing project helpers...")
from phase6_eval_prefix_quality import sample_validation_texts
from phase12_mamba2_head_permutation_equivalence import (
    MODEL_NAME,
    TOKENIZER_NAME,
    build_model,
    collect_mamba2_mixers,
    configure_mamba2_for_equivalence,
    ensure_local_hf_cache,
)
from phase13_mamba2_logical_head_prefix_quality import (
    build_head_orders,
    evaluate_model,
    replace_mixers_with_elastic,
)
from phase15_mamba2_anchor_head_prefix_eval import parse_anchor_layers
print("[phase16] imports finished")


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


STRATEGIES = [
    "random-22",
    "uniform-22",
    "anchor-22",
    "anchor-20",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--anchor_layers", type=str, default="0,1,5,15,20,21,22,23")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def resolve_strategies(requested_strategies):
    resolved = []
    seen = set()
    pattern = re.compile(r"^(random|uniform|anchor|uniform_grad_weight|anchor_grad_weight)-(\d+)$")
    for strategy in requested_strategies:
        match = pattern.match(strategy)
        if not match:
            raise ValueError(
                f"Unsupported strategy '{strategy}'. Expected one of "
                "random-<k>, uniform-<k>, anchor-<k>, uniform_grad_weight-<k>, anchor_grad_weight-<k>."
            )

        kind, heads_text = match.groups()
        active_heads = int(heads_text)
        if active_heads <= 0:
            raise ValueError(f"Strategy '{strategy}' must use positive active heads, got {active_heads}.")

        if kind == "uniform_grad_weight":
            canonical_kind = "uniform"
        elif kind == "anchor_grad_weight":
            canonical_kind = "anchor"
        else:
            canonical_kind = kind

        canonical_name = f"{canonical_kind}-{active_heads}"
        strategy_spec = {
            "requested": strategy,
            "name": canonical_name,
            "kind": canonical_kind,
            "active_heads": active_heads,
            "ordering": "random" if canonical_kind == "random" else "grad_weight",
        }

        if canonical_name not in seen:
            resolved.append(strategy_spec)
            seen.add(canonical_name)
    return resolved


@torch.no_grad()
def select_seeded_samples(all_samples, num_samples: int, seed: int):
    if len(all_samples) < num_samples:
        raise ValueError(f"Need {num_samples} samples, but only have {len(all_samples)} available")
    rng = random.Random(seed)
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)
    selected_indices = indices[:num_samples]
    return [all_samples[idx] for idx in selected_indices], selected_indices


@torch.no_grad()
def build_uniform_map(mixer_pairs, ordered_heads, active_heads: int):
    layer_active_heads_map = {}
    for (layer_idx, mixer), order in zip(mixer_pairs, ordered_heads):
        if active_heads > mixer.nheads:
            raise ValueError(f"active_heads={active_heads} exceeds nheads={mixer.nheads} at layer {layer_idx}")
        layer_active_heads_map[layer_idx] = torch.sort(order[:active_heads]).values.tolist()
    return layer_active_heads_map


@torch.no_grad()
def build_anchor_map(mixer_pairs, ordered_heads, anchor_layers, non_anchor_heads: int):
    anchor_layer_set = set(anchor_layers)
    layer_active_heads_map = {}
    for (layer_idx, mixer), order in zip(mixer_pairs, ordered_heads):
        if layer_idx in anchor_layer_set:
            layer_active_heads_map[layer_idx] = list(range(mixer.nheads))
        else:
            if non_anchor_heads > mixer.nheads:
                raise ValueError(
                    f"non_anchor_heads={non_anchor_heads} exceeds nheads={mixer.nheads} at layer {layer_idx}"
                )
            layer_active_heads_map[layer_idx] = torch.sort(order[:non_anchor_heads]).values.tolist()
    return layer_active_heads_map


@torch.no_grad()
def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)

    summary_rows = []
    metric_names = ["loss", "ppl", "logits_mse_to_full", "top1_agreement_to_full"]
    for strategy, group_rows in grouped.items():
        summary = {"row_type": "summary", "strategy": strategy, "seed": "", "anchor_layers": group_rows[0]["anchor_layers"]}
        summary["count"] = len(group_rows)
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in group_rows]
            summary[f"{metric_name}_mean"] = statistics.mean(values)
            summary[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return sorted(summary_rows, key=lambda row: row["strategy"])


@torch.no_grad()
def main():
    print("[phase16] entered main()")
    args = parse_args()
    anchor_layers = parse_anchor_layers(args.anchor_layers)
    strategies = resolve_strategies(args.strategies)

    csv_path = build_output_path(
        args.output_csv,
        "phase16_mamba2_multiseed_anchor_eval",
        tags={
            "seeds": "-".join(map(str, args.seeds)),
            "anchors": args.anchor_layers,
            "strategies": "-".join(args.strategies),
            "num_samples": args.num_samples,
            "seq_len": args.seq_len,
        },
    )
    ensure_local_hf_cache(csv_path.parent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    base_model, loader_name = build_model(device=device, dtype=dtype)
    print(f"Model loader: {loader_name}")
    base_model = configure_mamba2_for_equivalence(base_model, disable_mem_eff_path=True)
    mixer_pairs = collect_mamba2_mixers(base_model)
    num_layers = len(mixer_pairs)
    if num_layers != 24:
        print(f"[WARN] Expected 24 Mamba2 mixers, found {num_layers}")

    invalid_anchor_layers = [layer_idx for layer_idx in anchor_layers if layer_idx < 0 or layer_idx >= num_layers]
    if invalid_anchor_layers:
        raise ValueError(f"Invalid anchor layers: {invalid_anchor_layers}")

    candidate_pool_size = max(args.num_samples * 3, 1024)
    print("Loading WikiText-2 validation sample pool")
    all_samples = sample_validation_texts(tokenizer, num_samples=candidate_pool_size, seq_len=args.seq_len)

    raw_rows = []
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        eval_samples, eval_indices = select_seeded_samples(all_samples, args.num_samples, seed)
        calibration_count = min(8, len(eval_samples))
        calibration_samples = eval_samples[:calibration_count]

        print("Computing random head order")
        random_orders = build_head_orders(copy.deepcopy(base_model).eval(), "random", calibration_samples, device)
        print("Computing grad_weight head order")
        grad_orders = build_head_orders(copy.deepcopy(base_model).eval(), "grad_weight", calibration_samples, device)

        full_model = replace_mixers_with_elastic(base_model, layer_active_heads_map={})

        strategy_maps = {}
        for strategy in strategies:
            if strategy["kind"] == "random":
                layer_active_heads_map = build_uniform_map(mixer_pairs, random_orders, strategy["active_heads"])
            elif strategy["kind"] == "uniform":
                layer_active_heads_map = build_uniform_map(mixer_pairs, grad_orders, strategy["active_heads"])
            elif strategy["kind"] == "anchor":
                layer_active_heads_map = build_anchor_map(
                    mixer_pairs,
                    grad_orders,
                    anchor_layers,
                    strategy["active_heads"],
                )
            else:
                raise ValueError(f"Unsupported resolved strategy kind: {strategy['kind']}")
            strategy_maps[strategy["name"]] = layer_active_heads_map

        for strategy in strategies:
            strategy_name = strategy["name"]
            print(f"Evaluating strategy={strategy_name}")
            model = replace_mixers_with_elastic(base_model, layer_active_heads_map=strategy_maps[strategy_name])
            metrics = evaluate_model(model, full_model, eval_samples, device)
            print(
                f"  loss={metrics['next_token_loss']:.6f} ppl={metrics['perplexity']:.6f} "
                f"mse={metrics['logits_mse_to_full']:.8e} top1={metrics['top1_agreement_to_full']:.8f}"
            )
            raw_rows.append(
                {
                    "row_type": "raw",
                    "seed": seed,
                    "strategy": strategy_name,
                    "anchor_layers": str(anchor_layers),
                    "num_samples": args.num_samples,
                    "seq_len": args.seq_len,
                    "eval_sample_indices": str(eval_indices),
                    "loss": f"{metrics['next_token_loss']:.8f}",
                    "ppl": f"{metrics['perplexity']:.8f}",
                    "logits_mse_to_full": f"{metrics['logits_mse_to_full']:.8e}",
                    "top1_agreement_to_full": f"{metrics['top1_agreement_to_full']:.8f}",
                    "average_latency_ms": f"{metrics['average_latency_ms']:.6f}",
                }
            )

    summary_rows = summarize(raw_rows)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "row_type",
            "seed",
            "strategy",
            "anchor_layers",
            "num_samples",
            "seq_len",
            "eval_sample_indices",
            "loss",
            "ppl",
            "logits_mse_to_full",
            "top1_agreement_to_full",
            "average_latency_ms",
            "count",
            "loss_mean",
            "loss_std",
            "ppl_mean",
            "ppl_std",
            "logits_mse_to_full_mean",
            "logits_mse_to_full_std",
            "top1_agreement_to_full_mean",
            "top1_agreement_to_full_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(row)
        for row in summary_rows:
            writer.writerow(
                {
                    **row,
                    "loss_mean": f"{row['loss_mean']:.8f}",
                    "loss_std": f"{row['loss_std']:.8f}",
                    "ppl_mean": f"{row['ppl_mean']:.8f}",
                    "ppl_std": f"{row['ppl_std']:.8f}",
                    "logits_mse_to_full_mean": f"{row['logits_mse_to_full_mean']:.8e}",
                    "logits_mse_to_full_std": f"{row['logits_mse_to_full_std']:.8e}",
                    "top1_agreement_to_full_mean": f"{row['top1_agreement_to_full_mean']:.8f}",
                    "top1_agreement_to_full_std": f"{row['top1_agreement_to_full_std']:.8f}",
                }
            )

    print(f"Saved results to {csv_path}")
    print("\nMean +/- std summary")
    for row in summary_rows:
        print(
            f"strategy={row['strategy']} "
            f"loss={row['loss_mean']:.6f}+/-{row['loss_std']:.6f} "
            f"ppl={row['ppl_mean']:.6f}+/-{row['ppl_std']:.6f} "
            f"mse={row['logits_mse_to_full_mean']:.8e}+/-{row['logits_mse_to_full_std']:.8e} "
            f"top1={row['top1_agreement_to_full_mean']:.6f}+/-{row['top1_agreement_to_full_std']:.6f}"
        )


if __name__ == "__main__":
    main()
