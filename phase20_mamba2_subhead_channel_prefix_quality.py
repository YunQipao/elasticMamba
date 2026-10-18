import argparse
import copy
import csv
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from elastic_mamba2_subhead_block import ElasticMamba2SubheadBlock
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
    evaluate_model as evaluate_model_phase13,
    replace_mixers_with_elastic,
)
from phase15_mamba2_anchor_head_prefix_eval import build_anchor_map, parse_anchor_layers


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


DEFAULT_STRATEGIES = [
    "full",
    "head_only_anchor-22",
    "subhead_anchor-22-ch56",
    "subhead_anchor-22-ch48",
    "subhead_anchor-22-ch40",
    "subhead_anchor-21-ch56",
    "subhead_anchor-23-ch56",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--channel_keep", nargs="+", type=int, default=[40, 48, 56, 64])
    parser.add_argument("--non_anchor_heads", nargs="+", type=int, default=[21, 22, 23])
    parser.add_argument("--anchor_layers", type=str, default="0,1,5,15,20,21,22,23")
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--importance_csv", type=str, default="outputs/phase19_mamba2_intra_head_channel_importance.csv")
    parser.add_argument("--output_csv", type=str, default="outputs/phase20_mamba2_subhead_channel_prefix_quality.csv")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--sanity_check_phase15", action="store_true")
    return parser.parse_args()


def synchronize_if_needed(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def compute_latency_ms(model, input_ids, device: str):
    synchronize_if_needed(device)
    start = time.perf_counter()
    _ = model(input_ids)
    synchronize_if_needed(device)
    end = time.perf_counter()
    return (end - start) * 1000.0


def log(message: str):
    print(message, flush=True)


@torch.no_grad()
def replace_mixers_with_subhead_elastic(model, layer_config_map=None):
    if layer_config_map is None:
        layer_config_map = {}
    elastic_model = copy.deepcopy(model).eval()
    for layer_idx, layer in enumerate(elastic_model.backbone.layers):
        mixer = getattr(layer, "mixer", None)
        if mixer is not None and mixer.__class__.__name__ == "Mamba2":
            elastic = ElasticMamba2SubheadBlock.from_mamba2(mixer).eval()
            config = layer_config_map.get(layer_idx, {})
            elastic.set_active_head_indices(config.get("active_head_indices"))
            elastic.set_per_head_active_channel_indices(config.get("per_head_active_channel_indices"))
            layer.mixer = elastic
        elif isinstance(mixer, ElasticMamba2SubheadBlock):
            config = layer_config_map.get(layer_idx, {})
            mixer.set_active_head_indices(config.get("active_head_indices"))
            mixer.set_per_head_active_channel_indices(config.get("per_head_active_channel_indices"))
    return elastic_model


@torch.no_grad()
def evaluate_model(model, full_model, samples, device: str):
    total_loss = 0.0
    total_tokens = 0
    total_logits_mse = 0.0
    total_top1_match = 0.0
    total_positions = 0
    total_latency_ms = 0.0

    for sample_idx, input_ids in enumerate(samples, start=1):
        batch = input_ids.unsqueeze(0).to(device)
        logits = model(batch).logits
        full_logits = full_model(batch).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        token_count = shift_labels.numel()
        total_loss += loss.item()
        total_tokens += token_count

        total_logits_mse += torch.mean((logits.float() - full_logits.float()) ** 2).item()
        top1 = logits.argmax(dim=-1)
        full_top1 = full_logits.argmax(dim=-1)
        total_top1_match += (top1 == full_top1).float().sum().item()
        total_positions += top1.numel()
        total_latency_ms += compute_latency_ms(model, batch, device)
        if sample_idx == 1 or sample_idx % 16 == 0 or sample_idx == len(samples):
            log(f"  eval progress {sample_idx}/{len(samples)}")

    avg_loss = total_loss / total_tokens
    return {
        "next_token_loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "logits_mse_to_full": total_logits_mse / len(samples),
        "top1_agreement_to_full": total_top1_match / total_positions,
        "average_latency_ms": total_latency_ms / len(samples),
    }


def load_channel_importance_rankings(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Channel importance CSV not found: {csv_path}")

    grouped = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"layer_idx", "head_id", "channel_id", "rank_in_head"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Importance CSV missing required columns: {sorted(missing)}")
        for row in reader:
            layer_idx = int(row["layer_idx"])
            head_id = int(row["head_id"])
            channel_id = int(row["channel_id"])
            rank_in_head = int(row["rank_in_head"])
            grouped.setdefault((layer_idx, head_id), []).append((rank_in_head, channel_id))

    rankings = {}
    for key, items in grouped.items():
        items.sort(key=lambda item: item[0])
        rankings[key] = [channel_id for _, channel_id in items]
    return rankings


def check_required_inputs(importance_csv_path: Path):
    if not importance_csv_path.exists():
        raise FileNotFoundError(
            f"Required phase19 importance CSV not found: {importance_csv_path}. "
            "Please run phase19 first or pass --importance_csv."
        )
    if not importance_csv_path.is_file():
        raise ValueError(f"Importance CSV path is not a regular file: {importance_csv_path}")


def resolve_strategies(strategy_names, allowed_heads, allowed_channels):
    resolved = []
    skipped = []
    for name in strategy_names:
        if name == "full":
            resolved.append({"name": name, "kind": "full", "non_anchor_heads": 24, "channel_keep": 64})
            continue
        if name.startswith("head_only_anchor-"):
            non_anchor_heads = int(name.split("-")[-1])
            if non_anchor_heads not in allowed_heads:
                skipped.append(name)
                continue
            resolved.append(
                {
                    "name": name,
                    "kind": "head_only_anchor",
                    "non_anchor_heads": non_anchor_heads,
                    "channel_keep": 64,
                }
            )
            continue
        if name.startswith("subhead_anchor-"):
            parts = name.split("-")
            if len(parts) != 3 or not parts[2].startswith("ch"):
                raise ValueError(f"Unsupported strategy format: {name}")
            non_anchor_heads = int(parts[1])
            channel_keep = int(parts[2][2:])
            if non_anchor_heads not in allowed_heads or channel_keep not in allowed_channels:
                skipped.append(name)
                continue
            resolved.append(
                {
                    "name": name,
                    "kind": "subhead_anchor",
                    "non_anchor_heads": non_anchor_heads,
                    "channel_keep": channel_keep,
                }
            )
            continue
        raise ValueError(f"Unsupported strategy: {name}")
    if skipped:
        print(f"[INFO] Skipping incompatible strategies: {', '.join(skipped)}")
    return resolved


@torch.no_grad()
def build_strategy_layer_config_map(mixer_pairs, ordered_heads, rankings, anchor_layers, strategy):
    anchor_layer_set = set(anchor_layers)
    layer_config_map = {}

    for (layer_idx, mixer), order in zip(mixer_pairs, ordered_heads):
        if layer_idx in anchor_layer_set:
            active_heads = list(range(mixer.nheads))
            channel_keep = mixer.headdim
        else:
            active_heads = torch.sort(order[: strategy["non_anchor_heads"]]).values.tolist()
            channel_keep = strategy["channel_keep"]

        if channel_keep > mixer.headdim:
            raise ValueError(
                f"channel_keep={channel_keep} exceeds headdim={mixer.headdim} at layer {layer_idx}"
            )

        per_head_active_channel_indices = {}
        for head_id in active_heads:
            if channel_keep == mixer.headdim:
                selected_channels = list(range(mixer.headdim))
            else:
                key = (layer_idx, head_id)
                if key not in rankings:
                    raise ValueError(f"Missing importance ranking for layer={layer_idx} head={head_id}")
                selected_channels = sorted(rankings[key][:channel_keep])
            per_head_active_channel_indices[head_id] = selected_channels

        layer_config_map[layer_idx] = {
            "active_head_indices": active_heads,
            "per_head_active_channel_indices": per_head_active_channel_indices,
        }
    return layer_config_map


@torch.no_grad()
def compute_effective_total_channels(layer_config_map, mixer_pairs):
    total = 0
    for layer_idx, mixer in mixer_pairs:
        config = layer_config_map.get(layer_idx, {})
        active_heads = config.get("active_head_indices")
        if active_heads is None:
            active_heads = list(range(mixer.nheads))
        per_head = config.get("per_head_active_channel_indices") or {}
        for head_id in active_heads:
            selected = per_head.get(head_id)
            total += mixer.headdim if selected is None else len(selected)
    return total


def compute_budget_ratio(effective_total_channels, mixer_pairs):
    full_total_channels = sum(mixer.nheads * mixer.headdim for _, mixer in mixer_pairs)
    return effective_total_channels / full_total_channels


def verify_ch64_head_only_alignment(strategy_rows):
    by_name = {row["strategy"]: row for row in strategy_rows}
    target = by_name.get("head_only_anchor-22")
    reference = by_name.get("subhead_anchor-22-ch64")
    if target is None or reference is None:
        return

    fields = [
        "next_token_loss",
        "perplexity",
        "logits_mse_to_full",
        "top1_agreement_to_full",
        "effective_total_channels",
        "budget_ratio",
    ]
    for field in fields:
        lhs = float(target[field])
        rhs = float(reference[field])
        if abs(lhs - rhs) > 1e-6 * max(1.0, abs(lhs), abs(rhs)):
            raise RuntimeError(
                "subhead ch64 does not match head-only anchor-22; wrapper likely has a bug. "
                f"field={field} head_only={lhs} subhead_ch64={rhs}"
            )


def get_row_by_strategy(strategy_rows, strategy_name: str):
    for row in strategy_rows:
        if row["strategy"] == strategy_name:
            return row
    return None


def run_phase15_sanity_checks(base_model, mixer_pairs, ordered_heads, anchor_layers, samples, device: str, strategy_rows):
    raw_full_metrics = evaluate_model_phase13(base_model, base_model, samples, device)

    phase15_full_model = replace_mixers_with_elastic(base_model, layer_active_heads_map={})
    phase15_full_metrics = evaluate_model_phase13(phase15_full_model, phase15_full_model, samples, device)

    full_loss_diff = abs(raw_full_metrics["next_token_loss"] - phase15_full_metrics["next_token_loss"])
    if full_loss_diff > 1e-4:
        raise RuntimeError(
            "Sanity check failed: raw full loss does not match phase15 full loss within 1e-4. "
            f"raw_full_loss={raw_full_metrics['next_token_loss']:.8f} "
            f"phase15_full_loss={phase15_full_metrics['next_token_loss']:.8f} diff={full_loss_diff:.8f}"
        )

    phase15_anchor_map = build_anchor_map(mixer_pairs, ordered_heads, anchor_layers, 22)
    phase15_anchor_model = replace_mixers_with_elastic(base_model, layer_active_heads_map=phase15_anchor_map)
    phase15_anchor_metrics = evaluate_model_phase13(phase15_anchor_model, base_model, samples, device)

    head_only_row = get_row_by_strategy(strategy_rows, "head_only_anchor-22")
    if head_only_row is None:
        raise RuntimeError("Sanity check failed: missing strategy row 'head_only_anchor-22'")

    head_only_loss = float(head_only_row["next_token_loss"])
    head_only_ppl = float(head_only_row["perplexity"])
    loss_diff = abs(head_only_loss - phase15_anchor_metrics["next_token_loss"])
    ppl_diff = abs(head_only_ppl - phase15_anchor_metrics["perplexity"])
    if loss_diff > 1e-4 or ppl_diff > 1e-2:
        raise RuntimeError(
            "Sanity check failed: ch64 anchor-22 does not match phase15 anchor-22 closely enough. "
            f"phase20_loss={head_only_loss:.8f} phase15_loss={phase15_anchor_metrics['next_token_loss']:.8f} loss_diff={loss_diff:.8f}; "
            f"phase20_ppl={head_only_ppl:.8f} phase15_ppl={phase15_anchor_metrics['perplexity']:.8f} ppl_diff={ppl_diff:.8f}"
        )

    return raw_full_metrics, phase15_full_metrics, phase15_anchor_metrics


@torch.no_grad()
def main():
    args = parse_args()
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_local_hf_cache(output_csv_path.parent)
    importance_csv_path = Path(args.importance_csv)
    check_required_inputs(importance_csv_path)
    log(f"Using importance CSV: {importance_csv_path}")

    anchor_layers = parse_anchor_layers(args.anchor_layers)
    resolved_strategies = resolve_strategies(args.strategies, set(args.non_anchor_heads), set(args.channel_keep))
    if all(strategy["name"] != "head_only_anchor-22" for strategy in resolved_strategies):
        resolved_strategies.append(
            {"name": "head_only_anchor-22", "kind": "head_only_anchor", "non_anchor_heads": 22, "channel_keep": 64}
        )
    if all(strategy["name"] != "subhead_anchor-22-ch64" for strategy in resolved_strategies):
        resolved_strategies.append(
            {"name": "subhead_anchor-22-ch64", "kind": "subhead_anchor", "non_anchor_heads": 22, "channel_keep": 64}
        )

    rankings = load_channel_importance_rankings(importance_csv_path)
    log(f"Loaded channel rankings for {len(rankings)} layer/head pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    log(f"Loading tokenizer: {TOKENIZER_NAME} (local_files_only={args.local_files_only})")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=args.local_files_only)
    log(f"Loading model: {MODEL_NAME} (local_files_only={args.local_files_only})")
    base_model, loader_name = build_model(device=device, dtype=dtype, local_files_only=args.local_files_only)
    log(f"Model loader: {loader_name}")
    base_model = configure_mamba2_for_equivalence(base_model, disable_mem_eff_path=True)
    mixer_pairs = collect_mamba2_mixers(base_model)
    if len(mixer_pairs) != 24:
        log(f"[WARN] Expected 24 Mamba2 mixers, found {len(mixer_pairs)}")

    invalid_anchor_layers = [layer_idx for layer_idx in anchor_layers if layer_idx < 0 or layer_idx >= len(mixer_pairs)]
    if invalid_anchor_layers:
        raise ValueError(f"Invalid anchor layers: {invalid_anchor_layers}")

    log("Loading and tokenizing WikiText-2 validation samples")
    samples = sample_validation_texts(tokenizer, num_samples=args.num_samples, seq_len=args.seq_len)
    calibration_samples = samples[: min(8, len(samples))]
    log(f"Prepared {len(samples)} eval samples and {len(calibration_samples)} calibration samples")

    log("Computing grad_weight head order")
    ordered_heads = build_head_orders(copy.deepcopy(base_model).eval(), "grad_weight", calibration_samples, device)
    log(f"Computed grad_weight head order for {len(ordered_heads)} layers")

    raw_full_model = copy.deepcopy(base_model).eval()
    full_model = replace_mixers_with_subhead_elastic(base_model, layer_config_map={})
    log("Evaluating raw full model reference")
    raw_full_metrics = evaluate_model_phase13(raw_full_model, raw_full_model, samples, device)
    log(
        f"raw full loss={raw_full_metrics['next_token_loss']:.6f} ppl={raw_full_metrics['perplexity']:.6f} "
        f"latency_ms={raw_full_metrics['average_latency_ms']:.4f}"
    )
    log("Evaluating wrapped full strategy against raw full model")
    full_metrics = evaluate_model_phase13(full_model, raw_full_model, samples, device)
    log(
        f"full loss={full_metrics['next_token_loss']:.6f} ppl={full_metrics['perplexity']:.6f} "
        f"latency_ms={full_metrics['average_latency_ms']:.4f}"
    )
    rows = []

    for strategy in resolved_strategies:
        log(f"Evaluating strategy={strategy['name']}")
        layer_config_map = build_strategy_layer_config_map(
            mixer_pairs,
            ordered_heads,
            rankings,
            anchor_layers,
            strategy,
        )
        model = replace_mixers_with_subhead_elastic(base_model, layer_config_map=layer_config_map)
        metrics = evaluate_model_phase13(model, raw_full_model, samples, device)
        effective_total_channels = compute_effective_total_channels(layer_config_map, mixer_pairs)
        budget_ratio = compute_budget_ratio(effective_total_channels, mixer_pairs)
        if strategy["name"] == "full":
            full_metrics = metrics
        rows.append(
            {
                "strategy": strategy["name"],
                "anchor_layers": str(anchor_layers),
                "non_anchor_heads": strategy["non_anchor_heads"],
                "channel_keep": strategy["channel_keep"],
                "num_samples": args.num_samples,
                "seq_len": args.seq_len,
                "seed": args.seed,
                "next_token_loss": f"{metrics['next_token_loss']:.8f}",
                "perplexity": f"{metrics['perplexity']:.8f}",
                "logits_mse_to_full": f"{metrics['logits_mse_to_full']:.8e}",
                "top1_agreement_to_full": f"{metrics['top1_agreement_to_full']:.8f}",
                "average_latency_ms": f"{metrics['average_latency_ms']:.6f}",
                "effective_total_channels": effective_total_channels,
                "budget_ratio": f"{budget_ratio:.8f}",
                "full_loss": f"{raw_full_metrics['next_token_loss']:.8f}",
                "full_perplexity": f"{raw_full_metrics['perplexity']:.8f}",
            }
        )
        log(
            f"strategy={strategy['name']} loss={metrics['next_token_loss']:.6f} ppl={metrics['perplexity']:.6f} "
            f"mse={metrics['logits_mse_to_full']:.8e} top1={metrics['top1_agreement_to_full']:.8f} "
            f"latency_ms={metrics['average_latency_ms']:.4f} budget={budget_ratio:.6f}"
        )

    verify_ch64_head_only_alignment(rows)

    if args.sanity_check_phase15:
        log("Running sanity checks against phase15 evaluation path")
        raw_full_metrics_check, phase15_full_metrics, phase15_anchor_metrics = run_phase15_sanity_checks(
            base_model,
            mixer_pairs,
            ordered_heads,
            anchor_layers,
            samples,
            device,
            rows,
        )
        full_row = get_row_by_strategy(rows, "full")
        if full_row is None:
            raise RuntimeError("Sanity check failed: missing 'full' strategy row")
        full_row_loss = float(full_row["next_token_loss"])
        full_loss_diff = abs(full_row_loss - phase15_full_metrics["next_token_loss"])
        if full_loss_diff > 1e-4:
            raise RuntimeError(
                "Sanity check failed: phase20 full strategy loss does not match phase15 full loss within 1e-4. "
                f"phase20_full_loss={full_row_loss:.8f} phase15_full_loss={phase15_full_metrics['next_token_loss']:.8f} "
                f"diff={full_loss_diff:.8f}"
            )
        log(
            f"Sanity checks passed: raw_full_loss={raw_full_metrics_check['next_token_loss']:.8f} "
            f"phase15_full_loss={phase15_full_metrics['next_token_loss']:.8f} "
            f"phase15_anchor22_ppl={phase15_anchor_metrics['perplexity']:.8f}"
        )

    filtered_rows = [row for row in rows if row["strategy"] in set(args.strategies)]
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "anchor_layers",
                "non_anchor_heads",
                "channel_keep",
                "num_samples",
                "seq_len",
                "seed",
                "next_token_loss",
                "perplexity",
                "logits_mse_to_full",
                "top1_agreement_to_full",
                "average_latency_ms",
                "effective_total_channels",
                "budget_ratio",
                "full_loss",
                "full_perplexity",
            ],
        )
        writer.writeheader()
        writer.writerows(filtered_rows)
    log(f"Saved results to {output_csv_path}")


if __name__ == "__main__":
    main()
