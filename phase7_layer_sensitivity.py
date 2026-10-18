import argparse
import copy
import csv
from pathlib import Path

import torch
from transformers import AutoTokenizer
from output_naming import build_output_path

from elastic_mamba1_block import ElasticMamba1Block
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.5, 0.75])
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="fp32", choices=list(DEFAULT_DTYPES.keys()))
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def replace_single_layer_with_elastic(model, layer_idx: int, active_channels: int, force_prefix_path: bool = True):
    layer = model.backbone.layers[layer_idx]
    mixer = getattr(layer, "mixer", None)
    if not isinstance(mixer, MambaCompatibleTypes):
        raise TypeError(f"Layer {layer_idx} mixer is not Mamba-compatible: {type(mixer)}")
    if not isinstance(mixer, ElasticMamba1Block):
        elastic = ElasticMamba1Block.from_mamba(mixer).eval()
        elastic.set_active_channels(active_channels, force_prefix_path=force_prefix_path)
        layer.mixer = elastic
    else:
        mixer.set_active_channels(active_channels, force_prefix_path=force_prefix_path)
    return model


MambaCompatibleTypes = tuple()


def main():
    global MambaCompatibleTypes
    from mamba_ssm.modules.mamba_simple import Mamba

    MambaCompatibleTypes = (Mamba, ElasticMamba1Block)

    args = parse_args()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = prepare_runtime()
    dtype = DEFAULT_DTYPES[args.dtype]

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    base_model = build_model(device=device, dtype=dtype)

    print("Loading and tokenizing WikiText-2 validation samples")
    samples = sample_validation_texts(tokenizer, num_samples=args.num_samples, seq_len=args.seq_len)
    calibration_samples = samples[: min(8, len(samples))]

    print("Applying grad_weight ordering")
    ordered_model, _ = apply_ordering_to_model(base_model, "grad_weight", calibration_samples, device)
    full_model = copy.deepcopy(ordered_model).eval()
    full_metrics = evaluate_model(full_model, full_model, samples, device)

    mixer_pairs = collect_mamba_mixers(ordered_model)
    csv_path = build_output_path(
        args.output_csv,
        "phase7_layer_sensitivity",
        tags={
            "dtype": args.dtype,
            "ratios": "-".join(f"{ratio:.2f}" for ratio in args.ratios),
            "num_samples": args.num_samples,
            "seq_len": args.seq_len,
        },
    )

    rows = []
    for ratio in args.ratios:
        k = int(mixer_pairs[0][1].d_inner * ratio)
        print(f"Evaluating ratio={ratio:.2f}, k={k}")
        ratio_rows = []
        for layer_idx, _ in mixer_pairs:
            test_model = copy.deepcopy(ordered_model).eval()
            replace_single_layer_with_elastic(test_model, layer_idx=layer_idx, active_channels=k, force_prefix_path=True)
            metrics = evaluate_model(test_model, full_model, samples, device)
            loss_increase = metrics["next_token_loss"] - full_metrics["next_token_loss"]
            ratio_rows.append(
                {
                    "layer_idx": layer_idx,
                    "ratio": f"{ratio:.2f}",
                    "active_channels": k,
                    "loss": metrics["next_token_loss"],
                    "perplexity": metrics["perplexity"],
                    "loss_increase": loss_increase,
                    "average_latency_ms": metrics["average_latency_ms"],
                }
            )
            print(
                f"layer={layer_idx:02d} ratio={ratio:.2f} loss={metrics['next_token_loss']:.6f} "
                f"ppl={metrics['perplexity']:.6f} loss_increase={loss_increase:.6f}"
            )

        ratio_rows.sort(key=lambda row: row["loss_increase"], reverse=True)
        print(f"Top sensitive layers for ratio={ratio:.2f}: {[row['layer_idx'] for row in ratio_rows[:5]]}")
        rows.extend(ratio_rows)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_idx",
                "ratio",
                "active_channels",
                "loss",
                "perplexity",
                "loss_increase",
                "average_latency_ms",
                "full_loss",
                "full_perplexity",
                "num_samples",
                "seq_len",
                "dtype",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "layer_idx": row["layer_idx"],
                    "ratio": row["ratio"],
                    "active_channels": row["active_channels"],
                    "loss": f"{row['loss']:.8f}",
                    "perplexity": f"{row['perplexity']:.8f}",
                    "loss_increase": f"{row['loss_increase']:.8f}",
                    "average_latency_ms": f"{row['average_latency_ms']:.6f}",
                    "full_loss": f"{full_metrics['next_token_loss']:.8f}",
                    "full_perplexity": f"{full_metrics['perplexity']:.8f}",
                    "num_samples": args.num_samples,
                    "seq_len": args.seq_len,
                    "dtype": args.dtype,
                }
            )

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
