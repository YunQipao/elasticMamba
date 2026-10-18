import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
    prepare_runtime,
    ratio_to_active_channels,
    sample_validation_texts,
)
from mamba_ssm.modules.mamba_simple import Mamba


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


SEED = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_layers", type=str, default="0,18,19,22,23")
    parser.add_argument("--global_ratios", nargs="+", type=float, default=[0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="fp32", choices=list(DEFAULT_DTYPES.keys()))
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def synchronize_if_needed(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


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
def build_elastic_model(ordered_model, layer_ratio_map=None, force_prefix_path=True):
    if layer_ratio_map is None:
        layer_ratio_map = {}

    elastic_model = copy.deepcopy(ordered_model).eval()
    for layer_idx, layer in enumerate(elastic_model.backbone.layers):
        mixer = getattr(layer, "mixer", None)
        if isinstance(mixer, Mamba):
            elastic = ElasticMamba1Block.from_mamba(mixer).eval()
            ratio = layer_ratio_map.get(layer_idx)
            if ratio is not None:
                elastic.set_active_channels(
                    ratio_to_active_channels(elastic.d_inner, ratio),
                    force_prefix_path=force_prefix_path,
                )
            layer.mixer = elastic
        elif isinstance(mixer, ElasticMamba1Block):
            ratio = layer_ratio_map.get(layer_idx)
            if ratio is not None:
                mixer.set_active_channels(
                    ratio_to_active_channels(mixer.d_inner, ratio),
                    force_prefix_path=force_prefix_path,
                )
    return elastic_model


@torch.no_grad()
def compute_metrics_for_models(strategy_models, full_model, samples, device: str):
    strategy_names = list(strategy_models.keys())
    accumulators = {
        name: {
            "loss_sum": 0.0,
            "token_count": 0,
            "logits_mse_sum": 0.0,
            "latency_sum_ms": 0.0,
        }
        for name in strategy_names
    }
    full_loss_sum = 0.0
    full_token_count = 0

    for input_ids in samples:
        batch = input_ids.unsqueeze(0).to(device)
        labels = batch[:, 1:]

        full_logits = full_model(batch).logits
        full_shift_logits = full_logits[:, :-1, :]
        full_loss_sum += F.cross_entropy(
            full_shift_logits.reshape(-1, full_shift_logits.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        ).item()
        full_token_count += labels.numel()

        for name, model in strategy_models.items():
            synchronize_if_needed(device)
            start = time.perf_counter()
            logits = model(batch).logits
            synchronize_if_needed(device)
            end = time.perf_counter()

            shift_logits = logits[:, :-1, :]
            accumulators[name]["loss_sum"] += F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            ).item()
            accumulators[name]["token_count"] += labels.numel()
            accumulators[name]["logits_mse_sum"] += torch.mean((logits.float() - full_logits.float()) ** 2).item()
            accumulators[name]["latency_sum_ms"] += (end - start) * 1000.0

    full_loss = full_loss_sum / full_token_count
    full_perplexity = math.exp(full_loss)
    metrics = {}
    for name in strategy_names:
        accumulator = accumulators[name]
        loss = accumulator["loss_sum"] / accumulator["token_count"]
        perplexity = math.exp(loss)
        metrics[name] = {
            "next_token_loss": loss,
            "perplexity": perplexity,
            "logits_mse_to_full": accumulator["logits_mse_sum"] / len(samples),
            "average_latency_ms": accumulator["latency_sum_ms"] / len(samples),
        }
    return metrics, {"full_loss": full_loss, "full_perplexity": full_perplexity}


@torch.no_grad()
def layer_ratio_config_to_json(layer_ratio_map, d_inner):
    config = {}
    active_channels_map = {}
    actual_global_sum = 0.0
    for layer_idx in sorted(layer_ratio_map):
        ratio = layer_ratio_map[layer_idx]
        active_channels = ratio_to_active_channels(d_inner, ratio)
        actual_ratio = active_channels / d_inner
        config[str(layer_idx)] = round(actual_ratio, 8)
        active_channels_map[str(layer_idx)] = active_channels
        actual_global_sum += actual_ratio
    actual_global_ratio = actual_global_sum / len(layer_ratio_map) if layer_ratio_map else 0.0
    return json.dumps(config, sort_keys=True), json.dumps(active_channels_map, sort_keys=True), actual_global_ratio


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = prepare_runtime()
    dtype = DEFAULT_DTYPES[args.dtype]
    anchor_layers = parse_anchor_layers(args.anchor_layers)

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    base_model = build_model(device=device, dtype=dtype)
    mixer_pairs = collect_mamba_mixers(base_model)
    num_layers = len(mixer_pairs)
    if num_layers != 24:
        print(f"[WARN] Expected 24 Mamba mixers, found {num_layers}")

    invalid_anchor_layers = [layer_idx for layer_idx in anchor_layers if layer_idx < 0 or layer_idx >= num_layers]
    if invalid_anchor_layers:
        raise ValueError(f"Invalid anchor layer indices: {invalid_anchor_layers}")

    print("Loading and tokenizing WikiText-2 validation samples")
    samples = sample_validation_texts(tokenizer, num_samples=args.num_samples, seq_len=args.seq_len)
    calibration_samples = samples[: min(8, len(samples))]

    print("Applying grad_weight ordering")
    ordered_model, _ = apply_ordering_to_model(base_model, "grad_weight", calibration_samples, device)
    full_model = build_elastic_model(ordered_model, layer_ratio_map={}, force_prefix_path=False)

    csv_path = build_output_path(
        args.output_csv,
        "phase8_anchor_locking_eval",
        tags={
            "dtype": args.dtype,
            "global_ratios": "-".join(f"{ratio:.2f}" for ratio in args.global_ratios),
            "anchors": args.anchor_layers,
            "num_samples": args.num_samples,
            "seq_len": args.seq_len,
        },
    )

    d_inner = collect_mamba_mixers(ordered_model)[0][1].d_inner
    num_anchor = len(anchor_layers)
    num_non_anchor = num_layers - num_anchor
    rows = []

    for global_ratio in args.global_ratios:
        print(f"\nGlobal ratio={global_ratio:.2f}")
        uniform_ratio_map = {layer_idx: global_ratio for layer_idx in range(num_layers)}
        uniform_model = build_elastic_model(ordered_model, uniform_ratio_map, force_prefix_path=True)

        r_non_anchor = None
        anchor_ratio_map = None
        if num_non_anchor > 0:
            r_non_anchor = (global_ratio * num_layers - num_anchor) / num_non_anchor
            if r_non_anchor < 0.0 or r_non_anchor > 1.0:
                print(
                    f"  [SKIP] anchor locking invalid: r_non_anchor={r_non_anchor:.4f} out of [0, 1]"
                )
            else:
                anchor_ratio_map = {
                    layer_idx: (1.0 if layer_idx in anchor_layers else r_non_anchor)
                    for layer_idx in range(num_layers)
                }
        else:
            print("  [SKIP] no non-anchor layers available")

        strategy_models = {"uniform": uniform_model}
        if anchor_ratio_map is not None:
            anchor_model = build_elastic_model(ordered_model, anchor_ratio_map, force_prefix_path=True)
            strategy_models["anchor_locking"] = anchor_model

        metrics, full_metrics = compute_metrics_for_models(strategy_models, full_model, samples, device)
        uniform_metrics = metrics["uniform"]
        uniform_config_json, uniform_channels_json, uniform_actual_global_ratio = layer_ratio_config_to_json(
            uniform_ratio_map, d_inner
        )
        print(
            f"  uniform: loss={uniform_metrics['next_token_loss']:.6f} ppl={uniform_metrics['perplexity']:.6f} "
            f"latency_ms={uniform_metrics['average_latency_ms']:.4f}"
        )
        rows.append(
            {
                "global_ratio": f"{global_ratio:.2f}",
                "strategy": "uniform",
                "anchor_layers": json.dumps(anchor_layers),
                "num_anchor_layers": num_anchor,
                "num_layers": num_layers,
                "requested_global_ratio": f"{global_ratio:.4f}",
                "actual_global_ratio": f"{uniform_actual_global_ratio:.8f}",
                "r_non_anchor": "",
                "layer_ratio_config_json": uniform_config_json,
                "layer_active_channels_json": uniform_channels_json,
                "next_token_loss": f"{uniform_metrics['next_token_loss']:.8f}",
                "perplexity": f"{uniform_metrics['perplexity']:.8f}",
                "logits_mse_to_full": f"{uniform_metrics['logits_mse_to_full']:.8e}",
                "average_latency_ms": f"{uniform_metrics['average_latency_ms']:.6f}",
                "full_loss": f"{full_metrics['full_loss']:.8f}",
                "full_perplexity": f"{full_metrics['full_perplexity']:.8f}",
                "loss_delta_vs_uniform": "0.00000000",
                "status": "ok",
            }
        )

        if anchor_ratio_map is None:
            continue

        anchor_metrics = metrics["anchor_locking"]
        anchor_config_json, anchor_channels_json, anchor_actual_global_ratio = layer_ratio_config_to_json(
            anchor_ratio_map, d_inner
        )
        loss_delta_vs_uniform = uniform_metrics["next_token_loss"] - anchor_metrics["next_token_loss"]
        print(
            f"  anchor:  loss={anchor_metrics['next_token_loss']:.6f} ppl={anchor_metrics['perplexity']:.6f} "
            f"latency_ms={anchor_metrics['average_latency_ms']:.4f} loss_delta_vs_uniform={loss_delta_vs_uniform:.6f}"
        )
        rows.append(
            {
                "global_ratio": f"{global_ratio:.2f}",
                "strategy": "anchor_locking",
                "anchor_layers": json.dumps(anchor_layers),
                "num_anchor_layers": num_anchor,
                "num_layers": num_layers,
                "requested_global_ratio": f"{global_ratio:.4f}",
                "actual_global_ratio": f"{anchor_actual_global_ratio:.8f}",
                "r_non_anchor": f"{r_non_anchor:.8f}",
                "layer_ratio_config_json": anchor_config_json,
                "layer_active_channels_json": anchor_channels_json,
                "next_token_loss": f"{anchor_metrics['next_token_loss']:.8f}",
                "perplexity": f"{anchor_metrics['perplexity']:.8f}",
                "logits_mse_to_full": f"{anchor_metrics['logits_mse_to_full']:.8e}",
                "average_latency_ms": f"{anchor_metrics['average_latency_ms']:.6f}",
                "full_loss": f"{full_metrics['full_loss']:.8f}",
                "full_perplexity": f"{full_metrics['full_perplexity']:.8f}",
                "loss_delta_vs_uniform": f"{loss_delta_vs_uniform:.8f}",
                "status": "ok",
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_ratio",
                "strategy",
                "anchor_layers",
                "num_anchor_layers",
                "num_layers",
                "requested_global_ratio",
                "actual_global_ratio",
                "r_non_anchor",
                "layer_ratio_config_json",
                "layer_active_channels_json",
                "next_token_loss",
                "perplexity",
                "logits_mse_to_full",
                "average_latency_ms",
                "full_loss",
                "full_perplexity",
                "loss_delta_vs_uniform",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
