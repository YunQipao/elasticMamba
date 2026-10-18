import argparse
import copy
import csv
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer
from output_naming import build_output_path

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


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


SEED = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_layers", type=str, default="0,18,19,20,21,22,23")
    parser.add_argument("--non_anchor_heads", nargs="+", type=int, default=[20, 22])
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def parse_anchor_layers(anchor_layers: str):
    if not anchor_layers.strip():
        return []
    values = []
    for item in anchor_layers.split(","):
        token = item.strip()
        if token:
            values.append(int(token))
    return sorted(set(values))


@torch.no_grad()
def build_uniform_map(mixer_pairs, ordered_heads, active_heads: int):
    layer_active_heads_map = {}
    for (layer_idx, mixer), order in zip(mixer_pairs, ordered_heads):
        if active_heads > mixer.nheads:
            raise ValueError(f"active_heads={active_heads} exceeds nheads={mixer.nheads} at layer {layer_idx}")
        selected = torch.sort(order[:active_heads]).values.tolist()
        layer_active_heads_map[layer_idx] = selected
    return layer_active_heads_map


@torch.no_grad()
def build_anchor_map(mixer_pairs, ordered_heads, anchor_layers, non_anchor_heads: int):
    layer_active_heads_map = {}
    anchor_layer_set = set(anchor_layers)
    for (layer_idx, mixer), order in zip(mixer_pairs, ordered_heads):
        if layer_idx in anchor_layer_set:
            selected = list(range(mixer.nheads))
        else:
            if non_anchor_heads > mixer.nheads:
                raise ValueError(
                    f"non_anchor_heads={non_anchor_heads} exceeds nheads={mixer.nheads} at layer {layer_idx}"
                )
            selected = torch.sort(order[:non_anchor_heads]).values.tolist()
        layer_active_heads_map[layer_idx] = selected
    return layer_active_heads_map


@torch.no_grad()
def evaluate_strategy(base_model, full_model, samples, layer_active_heads_map, device: str):
    model = replace_mixers_with_elastic(base_model, layer_active_heads_map=layer_active_heads_map)
    return evaluate_model(model, full_model, samples, device)


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    csv_path = build_output_path(
        args.output_csv,
        "phase15_mamba2_anchor_head_prefix_eval",
        tags={
            "anchors": args.anchor_layers,
            "non_anchor_heads": "-".join(map(str, args.non_anchor_heads)),
            "num_samples": args.num_samples,
            "seq_len": args.seq_len,
        },
    )
    ensure_local_hf_cache(csv_path.parent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    anchor_layers = parse_anchor_layers(args.anchor_layers)

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

    print("Loading and tokenizing WikiText-2 validation samples")
    samples = sample_validation_texts(tokenizer, num_samples=args.num_samples, seq_len=args.seq_len)
    calibration_samples = samples[: min(8, len(samples))]

    print("Computing grad_weight head order")
    ordered_heads = build_head_orders(copy.deepcopy(base_model).eval(), "grad_weight", calibration_samples, device)

    full_model = replace_mixers_with_elastic(base_model, layer_active_heads_map={})
    full_metrics = evaluate_model(full_model, full_model, samples, device)
    print(
        f"full loss={full_metrics['next_token_loss']:.6f} ppl={full_metrics['perplexity']:.6f} "
        f"latency_ms={full_metrics['average_latency_ms']:.4f}"
    )

    rows = []
    for non_anchor_heads in args.non_anchor_heads:
        print(f"\n=== non_anchor_heads={non_anchor_heads} ===")
        uniform_map = build_uniform_map(mixer_pairs, ordered_heads, non_anchor_heads)
        anchor_map = build_anchor_map(mixer_pairs, ordered_heads, anchor_layers, non_anchor_heads)

        strategy_specs = [
            (f"uniform-{non_anchor_heads}", uniform_map),
            (f"anchor-{non_anchor_heads}", anchor_map),
        ]

        for strategy_name, layer_active_heads_map in strategy_specs:
            metrics = evaluate_strategy(base_model, full_model, samples, layer_active_heads_map, device)
            print(
                f"strategy={strategy_name} loss={metrics['next_token_loss']:.6f} "
                f"ppl={metrics['perplexity']:.6f} logits_mse={metrics['logits_mse_to_full']:.8e} "
                f"top1_agree={metrics['top1_agreement_to_full']:.8f} latency_ms={metrics['average_latency_ms']:.4f}"
            )
            rows.append(
                {
                    "strategy": strategy_name,
                    "anchor_layers": str(anchor_layers),
                    "non_anchor_heads": non_anchor_heads,
                    "loss": f"{metrics['next_token_loss']:.8f}",
                    "ppl": f"{metrics['perplexity']:.8f}",
                    "logits_mse_to_full": f"{metrics['logits_mse_to_full']:.8e}",
                    "top1_agreement_to_full": f"{metrics['top1_agreement_to_full']:.8f}",
                    "average_latency_ms": f"{metrics['average_latency_ms']:.6f}",
                    "full_loss": f"{full_metrics['next_token_loss']:.8f}",
                    "full_ppl": f"{full_metrics['perplexity']:.8f}",
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "anchor_layers",
                "non_anchor_heads",
                "loss",
                "ppl",
                "logits_mse_to_full",
                "top1_agreement_to_full",
                "average_latency_ms",
                "full_loss",
                "full_ppl",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
