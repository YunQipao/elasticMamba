import argparse
import copy
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer
from output_naming import build_output_path, build_related_output_path

from phase6_eval_prefix_quality import (
    DEFAULT_DTYPES,
    MODEL_NAME,
    TOKENIZER_NAME,
    apply_ordering_to_model,
    build_model,
    collect_mamba_mixers,
    evaluate_model,
    prepare_runtime,
    sample_validation_texts,
)
from phase8_anchor_locking_eval import build_elastic_model, parse_anchor_layers, ratio_to_active_channels


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


ANCHOR7 = [0, 18, 19, 20, 21, 22, 23]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.85, 0.90, 0.95, 1.0])
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="fp32", choices=list(DEFAULT_DTYPES.keys()))
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def select_seeded_samples(all_samples, num_samples: int, seed: int):
    if len(all_samples) < num_samples:
        raise ValueError(f"Need {num_samples} samples, but only have {len(all_samples)} available")
    rng = random.Random(seed)
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)
    selected = [all_samples[idx] for idx in indices[:num_samples]]
    return selected, indices[:num_samples]


@torch.no_grad()
def build_uniform_ratio_map(num_layers: int, ratio: float):
    return {layer_idx: ratio for layer_idx in range(num_layers)}


@torch.no_grad()
def build_anchor_ratio_map(num_layers: int, anchor_layers, global_ratio: float):
    num_anchor = len(anchor_layers)
    num_non_anchor = num_layers - num_anchor
    if num_non_anchor <= 0:
        return None, None
    r_non_anchor = (global_ratio * num_layers - num_anchor) / num_non_anchor
    if r_non_anchor < 0.0 or r_non_anchor > 1.0:
        return None, r_non_anchor
    ratio_map = {
        layer_idx: (1.0 if layer_idx in anchor_layers else r_non_anchor)
        for layer_idx in range(num_layers)
    }
    return ratio_map, r_non_anchor


@torch.no_grad()
def summarize_group(rows, group_key, metric_names):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in group_key)
        grouped[key].append(row)

    summaries = []
    for key, group_rows in grouped.items():
        summary = {name: value for name, value in zip(group_key, key)}
        summary["count"] = len(group_rows)
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in group_rows]
            mean_value = statistics.mean(values)
            std_value = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[f"{metric_name}_mean"] = mean_value
            summary[f"{metric_name}_std"] = std_value
        summaries.append(summary)
    return summaries


@torch.no_grad()
def build_summary_lookup(summary_rows):
    lookup = {}
    for row in summary_rows:
        key = (
            row["strategy"],
            row["order"],
            row["ratio"],
            row["dtype"],
            row["num_samples"],
            row["seq_len"],
        )
        lookup[key] = row
    return lookup


@torch.no_grad()
def compute_gain_rows(summary_rows):
    summary_lookup = build_summary_lookup(summary_rows)
    gain_rows = []

    anchor_rows = [row for row in summary_rows if row["strategy"] == "anchor7_grad_weight"]
    for anchor_row in anchor_rows:
        key_suffix = (
            anchor_row["ratio"],
            anchor_row["dtype"],
            anchor_row["num_samples"],
            anchor_row["seq_len"],
        )
        uniform_row = summary_lookup.get(("uniform_grad_weight", "grad_weight", *key_suffix))
        random_row = summary_lookup.get(("random_ordering", "random", *key_suffix))

        anchor_gain_vs_uniform_loss = ""
        anchor_gain_vs_uniform_ppl = ""
        anchor_gain_vs_random_ppl = ""

        if uniform_row is not None:
            anchor_gain_vs_uniform_loss = (
                uniform_row["next_token_loss_mean"] - anchor_row["next_token_loss_mean"]
            )
            anchor_gain_vs_uniform_ppl = (
                uniform_row["perplexity_mean"] - anchor_row["perplexity_mean"]
            )
        if random_row is not None:
            anchor_gain_vs_random_ppl = (
                random_row["perplexity_mean"] - anchor_row["perplexity_mean"]
            )

        gain_rows.append(
            {
                "ratio": anchor_row["ratio"],
                "dtype": anchor_row["dtype"],
                "num_samples": anchor_row["num_samples"],
                "seq_len": anchor_row["seq_len"],
                "anchor_gain_vs_uniform_loss": anchor_gain_vs_uniform_loss,
                "anchor_gain_vs_uniform_ppl": anchor_gain_vs_uniform_ppl,
                "anchor_gain_vs_random_ppl": anchor_gain_vs_random_ppl,
            }
        )

    return gain_rows


def main():
    args = parse_args()
    device = prepare_runtime()
    dtype = DEFAULT_DTYPES[args.dtype]
    anchor_layers = parse_anchor_layers(",".join(str(layer_idx) for layer_idx in ANCHOR7))

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    base_model = build_model(device=device, dtype=dtype)
    mixer_pairs = collect_mamba_mixers(base_model)
    num_layers = len(mixer_pairs)
    if num_layers != 24:
        print(f"[WARN] Expected 24 Mamba mixers, found {num_layers}")

    max_required = max(args.num_samples, 8)
    candidate_pool_size = max(args.num_samples * 3, 1024)
    print("Loading WikiText-2 validation sample pool")
    all_samples = sample_validation_texts(tokenizer, num_samples=candidate_pool_size, seq_len=args.seq_len)
    if len(all_samples) < max_required:
        raise ValueError(f"Need at least {max_required} tokenized samples, got {len(all_samples)}")

    raw_csv_path = build_output_path(
        args.output_csv,
        "phase10_multiseed_raw",
        tags={
            "dtype": args.dtype,
            "seeds": "-".join(map(str, args.seeds)),
            "ratios": "-".join(f"{ratio:.2f}" for ratio in args.ratios),
            "num_samples": args.num_samples,
            "seq_len": args.seq_len,
        },
    )
    summary_csv_path = build_related_output_path(raw_csv_path, "summary")
    gain_summary_csv_path = build_related_output_path(raw_csv_path, "gain_summary")

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

        strategies = [
            ("uniform_grad_weight", "grad_weight"),
            ("anchor7_grad_weight", "grad_weight"),
            ("random_ordering", "random"),
        ]

        for strategy_name, order_name in strategies:
            print(f"Applying strategy={strategy_name} order={order_name}")
            ordered_model, _ = apply_ordering_to_model(base_model, order_name, calibration_samples, device)
            full_model = build_elastic_model(ordered_model, layer_ratio_map={}, force_prefix_path=False)
            full_d_inner = collect_mamba_mixers(full_model)[0][1].d_inner

            for ratio in args.ratios:
                if strategy_name == "uniform_grad_weight" or strategy_name == "random_ordering":
                    ratio_map = build_uniform_ratio_map(num_layers, ratio)
                    extra_info = {"r_non_anchor": "", "anchor_layers": "[]"}
                else:
                    ratio_map, r_non_anchor = build_anchor_ratio_map(num_layers, anchor_layers, ratio)
                    if ratio_map is None:
                        print(f"  [SKIP] ratio={ratio:.2f} gives invalid r_non_anchor={r_non_anchor:.6f}")
                        continue
                    extra_info = {
                        "r_non_anchor": f"{r_non_anchor:.8f}",
                        "anchor_layers": str(anchor_layers),
                    }

                strategy_model = build_elastic_model(ordered_model, layer_ratio_map=ratio_map, force_prefix_path=True)
                metrics = evaluate_model(strategy_model, full_model, eval_samples, device)
                print(
                    f"  ratio={ratio:.2f} loss={metrics['next_token_loss']:.6f} "
                    f"ppl={metrics['perplexity']:.6f} logits_mse={metrics['logits_mse_to_full']:.8e} "
                    f"latency_ms={metrics['average_latency_ms']:.4f}"
                )

                raw_rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy_name,
                        "order": order_name,
                        "ratio": f"{ratio:.2f}",
                        "num_layers": num_layers,
                        "num_samples": args.num_samples,
                        "seq_len": args.seq_len,
                        "dtype": args.dtype,
                        "active_channels": ratio_to_active_channels(full_d_inner, ratio),
                        "eval_sample_indices": str(eval_indices),
                        "next_token_loss": f"{metrics['next_token_loss']:.8f}",
                        "perplexity": f"{metrics['perplexity']:.8f}",
                        "logits_mse": f"{metrics['logits_mse_to_full']:.8e}",
                        "average_latency_ms": f"{metrics['average_latency_ms']:.6f}",
                        **extra_info,
                    }
                )

    with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "strategy",
                "order",
                "ratio",
                "num_layers",
                "num_samples",
                "seq_len",
                "dtype",
                "active_channels",
                "eval_sample_indices",
                "next_token_loss",
                "perplexity",
                "logits_mse",
                "average_latency_ms",
                "r_non_anchor",
                "anchor_layers",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows = summarize_group(
        raw_rows,
        group_key=["strategy", "order", "ratio", "dtype", "num_samples", "seq_len"],
        metric_names=["next_token_loss", "perplexity", "logits_mse", "average_latency_ms"],
    )
    gain_rows = compute_gain_rows(summary_rows)
    gain_lookup = {
        (row["ratio"], row["dtype"], row["num_samples"], row["seq_len"]): row
        for row in gain_rows
    }

    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "order",
                "ratio",
                "dtype",
                "num_samples",
                "seq_len",
                "count",
                "next_token_loss_mean",
                "next_token_loss_std",
                "perplexity_mean",
                "perplexity_std",
                "logits_mse_mean",
                "logits_mse_std",
                "average_latency_ms_mean",
                "average_latency_ms_std",
                "anchor_gain_vs_uniform_loss",
                "anchor_gain_vs_uniform_ppl",
                "anchor_gain_vs_random_ppl",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            gain_row = gain_lookup.get((row["ratio"], row["dtype"], row["num_samples"], row["seq_len"]), {})
            writer.writerow(
                {
                    **{k: row[k] for k in ["strategy", "order", "ratio", "dtype", "num_samples", "seq_len", "count"]},
                    "next_token_loss_mean": f"{row['next_token_loss_mean']:.8f}",
                    "next_token_loss_std": f"{row['next_token_loss_std']:.8f}",
                    "perplexity_mean": f"{row['perplexity_mean']:.8f}",
                    "perplexity_std": f"{row['perplexity_std']:.8f}",
                    "logits_mse_mean": f"{row['logits_mse_mean']:.8e}",
                    "logits_mse_std": f"{row['logits_mse_std']:.8e}",
                    "average_latency_ms_mean": f"{row['average_latency_ms_mean']:.6f}",
                    "average_latency_ms_std": f"{row['average_latency_ms_std']:.6f}",
                    "anchor_gain_vs_uniform_loss": (
                        "" if gain_row.get("anchor_gain_vs_uniform_loss", "") == ""
                        else f"{gain_row['anchor_gain_vs_uniform_loss']:.8f}"
                    ),
                    "anchor_gain_vs_uniform_ppl": (
                        "" if gain_row.get("anchor_gain_vs_uniform_ppl", "") == ""
                        else f"{gain_row['anchor_gain_vs_uniform_ppl']:.8f}"
                    ),
                    "anchor_gain_vs_random_ppl": (
                        "" if gain_row.get("anchor_gain_vs_random_ppl", "") == ""
                        else f"{gain_row['anchor_gain_vs_random_ppl']:.8f}"
                    ),
                }
            )

    with gain_summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ratio",
                "dtype",
                "num_samples",
                "seq_len",
                "anchor_gain_vs_uniform_loss",
                "anchor_gain_vs_uniform_ppl",
                "anchor_gain_vs_random_ppl",
            ],
        )
        writer.writeheader()
        for row in gain_rows:
            writer.writerow(
                {
                    "ratio": row["ratio"],
                    "dtype": row["dtype"],
                    "num_samples": row["num_samples"],
                    "seq_len": row["seq_len"],
                    "anchor_gain_vs_uniform_loss": (
                        "" if row["anchor_gain_vs_uniform_loss"] == ""
                        else f"{row['anchor_gain_vs_uniform_loss']:.8f}"
                    ),
                    "anchor_gain_vs_uniform_ppl": (
                        "" if row["anchor_gain_vs_uniform_ppl"] == ""
                        else f"{row['anchor_gain_vs_uniform_ppl']:.8f}"
                    ),
                    "anchor_gain_vs_random_ppl": (
                        "" if row["anchor_gain_vs_random_ppl"] == ""
                        else f"{row['anchor_gain_vs_random_ppl']:.8f}"
                    ),
                }
            )

    print(f"Saved raw results to {raw_csv_path}")
    print(f"Saved summary results to {summary_csv_path}")
    print(f"Saved gain summary to {gain_summary_csv_path}")
    print("\nMean/std summary")
    for row in summary_rows:
        print(
            f"strategy={row['strategy']} ratio={row['ratio']} "
            f"loss={row['next_token_loss_mean']:.6f}+/-{row['next_token_loss_std']:.6f} "
            f"ppl={row['perplexity_mean']:.6f}+/-{row['perplexity_std']:.6f} "
            f"mse={row['logits_mse_mean']:.8e}+/-{row['logits_mse_std']:.8e} "
            f"latency_ms={row['average_latency_ms_mean']:.4f}+/-{row['average_latency_ms_std']:.4f}"
        )


if __name__ == "__main__":
    main()
