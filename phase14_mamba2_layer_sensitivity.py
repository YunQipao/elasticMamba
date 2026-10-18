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
TOP_K = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active_heads", nargs="+", type=int, default=[20, 22])
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    csv_path = build_output_path(
        args.output_csv,
        "phase14_mamba2_layer_sensitivity",
        tags={
            "active_heads": "-".join(map(str, args.active_heads)),
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
    for active_heads in args.active_heads:
        print(f"\n=== active_heads={active_heads} ===")
        per_head_rows = []
        for (layer_idx, mixer), order in zip(mixer_pairs, ordered_heads):
            if active_heads > mixer.nheads:
                raise ValueError(f"active_heads={active_heads} exceeds nheads={mixer.nheads} at layer {layer_idx}")

            selected = torch.sort(order[:active_heads]).values.tolist()
            layer_active_heads_map = {layer_idx: selected}
            model = replace_mixers_with_elastic(base_model, layer_active_heads_map=layer_active_heads_map)
            metrics = evaluate_model(model, full_model, samples, device)
            loss_increase = metrics["next_token_loss"] - full_metrics["next_token_loss"]

            row = {
                "layer_idx": layer_idx,
                "active_heads": active_heads,
                "loss": metrics["next_token_loss"],
                "ppl": metrics["perplexity"],
                "loss_increase_vs_full": loss_increase,
                "logits_mse_to_full": metrics["logits_mse_to_full"],
                "top1_agreement_to_full": metrics["top1_agreement_to_full"],
            }
            per_head_rows.append(row)
            rows.append(
                {
                    "layer_idx": layer_idx,
                    "active_heads": active_heads,
                    "loss": f"{metrics['next_token_loss']:.8f}",
                    "ppl": f"{metrics['perplexity']:.8f}",
                    "loss_increase_vs_full": f"{loss_increase:.8f}",
                    "logits_mse_to_full": f"{metrics['logits_mse_to_full']:.8e}",
                    "top1_agreement_to_full": f"{metrics['top1_agreement_to_full']:.8f}",
                }
            )
            print(
                f"layer={layer_idx} active_heads={active_heads} loss={metrics['next_token_loss']:.6f} "
                f"ppl={metrics['perplexity']:.6f} loss_increase={loss_increase:.6f} "
                f"mse={metrics['logits_mse_to_full']:.8e} top1={metrics['top1_agreement_to_full']:.8f}"
            )

        per_head_rows.sort(key=lambda row: row["loss_increase_vs_full"], reverse=True)
        print(f"Top-{min(TOP_K, len(per_head_rows))} sensitive layers for active_heads={active_heads}")
        for row in per_head_rows[:TOP_K]:
            print(
                f"  layer={row['layer_idx']} loss_increase={row['loss_increase_vs_full']:.6f} "
                f"loss={row['loss']:.6f} ppl={row['ppl']:.6f}"
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_idx",
                "active_heads",
                "loss",
                "ppl",
                "loss_increase_vs_full",
                "logits_mse_to_full",
                "top1_agreement_to_full",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
