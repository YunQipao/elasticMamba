from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from phase12_mamba2_head_permutation_equivalence import (
    MODEL_NAME,
    TOKENIZER_NAME,
    build_model,
    collect_mamba2_mixers,
    configure_mamba2_for_equivalence,
    ensure_local_hf_cache,
)
from phase21_mamba2_state_cache_permutation_equivalence import build_inference_params, synchronize_if_cuda
from phase22_mamba2_state_channel_importance_probe import (
    clone_cache_entry,
    extract_state_norms,
    pack_inference_params,
    restore_inference_params,
)


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


DEFAULT_STRATEGIES = [
    "full_state",
    "random_drop",
    "uniform_drop",
    "state_norm_keep",
    "layer_aware_statecache",
    "head_aware_statecache",
    "group_aware_statecache",
    "hybrid_norm_layer",
    "hybrid_norm_head",
    "hybrid_norm_group",
    "hybrid_norm_layer_head",
    "hybrid_norm_layer_head_group",
    "anchor_protected_state_norm",
]
DEFAULT_BUDGET_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9]
GROUP_SIZE = 8
DEFAULT_OUTPUT_STEM = "phase23_mamba2_state_cache_compression_eval"

CSV_FIELDNAMES = [
    "seed",
    "dataset",
    "strategy",
    "budget_ratio",
    "target_budget_ratio",
    "effective_budget_ratio",
    "budget_abs_error",
    "prompt_id",
    "prefix_len",
    "continuation_len",
    "full_loss",
    "continuation_loss",
    "loss_increase",
    "full_ppl",
    "continuation_ppl",
    "logits_mse_to_full",
    "top1_agreement_to_full",
    "original_state_elements",
    "total_cache_elements",
    "manageable_state_elements",
    "kept_manageable_elements",
    "unmanaged_elements",
    "kept_state_elements",
    "estimated_state_memory_bytes",
    "memory_reduction_ratio",
    "average_decode_latency_ms",
    "tokens_per_second",
    "clean_cache_changed",
    "alias_check_pass",
    "nested_budget_check",
    "priority_direction_check",
    "full_repeat_check",
    "keep_set_hash",
    "strategy_order",
    "hybrid_alpha",
    "anchor_min_keep_ratio",
    "norm_score",
    "sensitivity_score",
    "final_priority_score",
    "sensitivity_valid",
    "sensitivity_nonzero_count",
    "sensitivity_max",
    "sensitivity_mean_positive",
    "sensitivity_source_file",
    "sensitivity_score_std",
    "final_priority_changed_vs_norm",
    "kept_mean_priority",
    "dropped_mean_priority",
    "priority_top10",
    "priority_bottom10",
    "status",
    "error_msg",
]

BEST_STATECACHE_PRESET = [
    "full_state",
    "random_drop",
    "uniform_drop",
    "state_norm_keep",
    "hybrid_norm_head",
    "hybrid_norm_group",
]
SUMMARY_OUTPUT_PATH = Path("outputs/phase23_statecache_summary.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "synthetic"])
    parser.add_argument("--preset", type=str, default=None, choices=["best_statecache"])
    parser.add_argument("--num_prompts", type=int, default=64)
    parser.add_argument("--prefix_len", type=int, default=512)
    parser.add_argument("--continuation_len", type=int, default=64)
    parser.add_argument("--budget_ratios", nargs="+", type=float, default=DEFAULT_BUDGET_RATIOS)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--hybrid_alpha", nargs="+", type=float, default=[1.0])
    parser.add_argument("--phase22_layer_csv", type=str, default="outputs/phase22_latest_layer_sensitivity.csv")
    parser.add_argument("--phase22_head_csv", type=str, default="outputs/phase22_latest_head_sensitivity.csv")
    parser.add_argument("--phase22_group_csv", type=str, default="outputs/phase22_latest_group_sensitivity.csv")
    parser.add_argument("--anchor_layers", type=str, default="21,20,19,5,14,18")
    parser.add_argument("--anchor_heads", type=str, default="21:15,20:19,14:1,20:16,19:17,5:17")
    parser.add_argument("--anchor_min_keep_ratio", type=float, default=0.95)
    parser.add_argument("--anchor_mode", type=str, default="soft", choices=["hard", "soft"])
    parser.add_argument("--anchor_boost", nargs="+", type=float, default=[0.5])
    parser.add_argument("--cache_device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--disable_mem_eff_path", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--dry_run_plan", action="store_true")
    parser.add_argument("--dry_run_samples", action="store_true")
    parser.add_argument("--append_each_prompt", action="store_true")
    parser.add_argument("--debug_budget", action="store_true")
    parser.add_argument("--debug_cache_alias", action="store_true")
    parser.add_argument("--strategy_order", type=str, default="default", choices=["default", "reversed", "shuffled"])
    parser.add_argument("--validate_nested_budgets", action="store_true")
    parser.add_argument("--validate_full_repeat", action="store_true")
    parser.add_argument("--single_strategy_debug", action="store_true")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--fast_smoke", action="store_true")
    return parser.parse_args()


def normalize_strategies(strategies):
    normalized = []
    for strategy in strategies:
        if strategy not in DEFAULT_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")
        if strategy not in normalized:
            normalized.append(strategy)
    return normalized


def resolve_preset(args):
    if args.preset == "best_statecache":
        if args.strategies == DEFAULT_STRATEGIES:
            args.strategies = list(BEST_STATECACHE_PRESET)
        if args.hybrid_alpha == [1.0]:
            args.hybrid_alpha = [0.0, 0.25, 0.5, 1.0]
    return args


def resolve_seed_list(args):
    if args.seeds:
        seed_list = [int(seed) for seed in args.seeds]
    else:
        seed_list = [int(args.seed)]
    deduped = []
    for seed in seed_list:
        if seed not in deduped:
            deduped.append(seed)
    return deduped


def load_csv_rows(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader)
    except Exception:
        return []


def summarize_phase22_scores(rows, value_field: str, key_fields: list[str]):
    valid = []
    for row in rows:
        if value_field not in row:
            continue
        try:
            score = float(row[value_field])
        except Exception:
            continue
        if math.isnan(score):
            continue
        if score <= 0:
            continue
        key = tuple(int(row[field]) for field in key_fields if field in row)
        valid.append((key, score, row))
    valid.sort(key=lambda item: item[1], reverse=True)
    return valid


def parse_bool_like(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_phase22_sensitivity_file(path_str: str, rows, expected_level: str):
    warnings = []
    path = Path(path_str)
    info = {
        "path": str(path),
        "exists": path.exists(),
        "expected_level": expected_level,
        "valid": False,
        "positive_loss_count": 0,
        "max_loss_increase": 0.0,
        "mean_positive_loss_increase": 0.0,
        "nonzero_count": 0,
        "warning": "",
    }
    if not path.exists():
        info["warning"] = "missing_sensitivity_file"
        warnings.append(f"[WARN] missing sensitivity file for level={expected_level}: {path}")
        return info, warnings
    if not rows:
        info["warning"] = "empty_sensitivity_file"
        warnings.append(f"[WARN] empty sensitivity file for level={expected_level}: {path}")
        return info, warnings
    sample = rows[0]
    if "loss_increase" not in sample:
        info["warning"] = "missing_loss_increase_column"
        warnings.append(f"[WARN] sensitivity file missing loss_increase for level={expected_level}: {path}")
        return info, warnings
    levels = {row.get("ablation_level", "") for row in rows if row.get("ablation_level", "")}
    if levels and levels != {expected_level}:
        info["warning"] = "ablation_level_mismatch"
        warnings.append(f"[WARN] sensitivity file ablation_level mismatch for level={expected_level}: {path} found={sorted(levels)}")
        return info, warnings
    positive_count = 0
    max_loss = 0.0
    positives = []
    nonzero_count = 0
    for row in rows:
        try:
            value = float(row.get("loss_increase", 0.0))
        except Exception:
            continue
        if not math.isfinite(value):
            continue
        if value != 0.0:
            nonzero_count += 1
        if value > 0.0:
            positive_count += 1
            positives.append(value)
            max_loss = max(max_loss, value)
    info["positive_loss_count"] = positive_count
    info["max_loss_increase"] = max_loss
    info["mean_positive_loss_increase"] = (sum(positives) / len(positives)) if positives else 0.0
    info["nonzero_count"] = nonzero_count
    if "positive_loss_count" in sample:
        try:
            meta_positive = max(int(float(row.get("positive_loss_count", 0))) for row in rows)
            positive_count = max(positive_count, meta_positive)
            info["positive_loss_count"] = positive_count
        except Exception:
            pass
    if "max_loss_increase" in sample:
        try:
            meta_max = max(float(row.get("max_loss_increase", 0.0)) for row in rows)
            max_loss = max(max_loss, meta_max)
            info["max_loss_increase"] = max_loss
        except Exception:
            pass
    if "sensitivity_valid" in sample and any(parse_bool_like(row.get("sensitivity_valid", False)) for row in rows):
        pass
    if positive_count <= 0 or max_loss <= 0.0:
        info["warning"] = "all_zero_loss_increase"
        warnings.append(f"[WARN] invalid sensitivity signal for level={expected_level}: positive_loss_count={positive_count} max_loss_increase={max_loss} file={path}")
        return info, warnings
    info["valid"] = True
    return info, warnings


def parse_phase22_tables(layer_csv: str, head_csv: str, group_csv: str):
    layer_rows = load_csv_rows(layer_csv)
    head_rows = load_csv_rows(head_csv)
    group_rows = load_csv_rows(group_csv)
    layer_info, layer_warnings = validate_phase22_sensitivity_file(layer_csv, layer_rows, "layer")
    head_info, head_warnings = validate_phase22_sensitivity_file(head_csv, head_rows, "head")
    group_info, group_warnings = validate_phase22_sensitivity_file(group_csv, group_rows, "group")
    for warning in layer_warnings + head_warnings + group_warnings:
        print(warning)
    layer_scores = summarize_phase22_scores(layer_rows, "loss_increase", ["layer_idx"]) if layer_info["valid"] else []
    head_scores = summarize_phase22_scores(head_rows, "loss_increase", ["layer_idx", "head_id"]) if head_info["valid"] else []
    group_scores = summarize_phase22_scores(group_rows, "loss_increase", ["layer_idx", "head_id", "channel_group_id"]) if group_info["valid"] else []
    return {
        "layer_rows": layer_rows,
        "head_rows": head_rows,
        "group_rows": group_rows,
        "layer_scores": layer_scores,
        "head_scores": head_scores,
        "group_scores": group_scores,
        "layer_info": layer_info,
        "head_info": head_info,
        "group_info": group_info,
    }


def print_dry_run_plan(args, score_info):
    expanded = expand_strategy_variants(args.strategies, args.hybrid_alpha, args.anchor_boost)
    seed_list = resolve_seed_list(args)
    estimated_runs = len(expanded) * len(args.budget_ratios) * args.num_prompts * len(seed_list)
    print("phase23 dry run plan")
    print(f"  preset                 : {args.preset}")
    print(f"  dataset                : {args.dataset}")
    print(f"  seeds                  : {seed_list}")
    print(f"  strategies             : {args.strategies}")
    print(f"  expanded strategies    : {expanded}")
    print(f"  hybrid_alpha           : {args.hybrid_alpha}")
    print(f"  anchor_mode            : {args.anchor_mode}")
    print(f"  anchor_boost           : {args.anchor_boost}")
    print(f"  anchor_layers          : {args.anchor_layers}")
    print(f"  anchor_heads           : {args.anchor_heads}")
    print(f"  anchor_min_keep_ratio  : {args.anchor_min_keep_ratio}")
    print(f"  budget_ratios          : {args.budget_ratios}")
    print(f"  estimated runs         : {estimated_runs}")
    print(f"  phase22 layer csv exists: {Path(args.phase22_layer_csv).exists()}")
    print(f"  phase22 head csv exists : {Path(args.phase22_head_csv).exists()}")
    print(f"  phase22 group csv exists: {Path(args.phase22_group_csv).exists()}")
    print(f"  valid layer scores      : {len(score_info['layer_scores'])}")
    print(f"  valid head scores       : {len(score_info['head_scores'])}")
    print(f"  valid group scores      : {len(score_info['group_scores'])}")


def sample_token_chunks(tokenizer, dataset_name: str, split: str, num_prompts: int, prefix_len: int, continuation_len: int, seed: int):
    total_len = prefix_len + continuation_len
    if dataset_name == "synthetic":
        return build_synthetic_eval_samples(tokenizer, num_prompts, prefix_len, continuation_len, seed)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("`datasets` is required for wikitext2 sampling") from exc

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = []
    for item in dataset:
        text = item["text"].strip()
        if text:
            texts.append(text)
    if not texts:
        raise RuntimeError("No non-empty texts found in WikiText-2 validation split")

    joined_text = "\n\n".join(texts)
    encoded = tokenizer(joined_text, return_tensors="pt", add_special_tokens=False)
    all_ids = encoded["input_ids"][0].long()
    total_token_count = int(all_ids.numel())
    print(f"total token count: {total_token_count}")
    print(f"requested total_len: {total_len}")
    if total_token_count < total_len + 1:
        raise RuntimeError(f"Insufficient tokens: token_count={total_token_count}, required_at_least={total_len + 1}")

    max_start = total_token_count - total_len
    generator = random.Random(seed)
    starts = list(range(max_start + 1))
    generator.shuffle(starts)

    selected = []
    for start in starts:
        chunk = all_ids[start:start + total_len]
        if chunk.numel() == total_len:
            selected.append(
                {
                    "prompt_id": len(selected),
                    "dataset": dataset_name,
                    "input_ids": chunk.view(1, total_len).clone(),
                    "prefix_ids": chunk[:prefix_len].view(1, prefix_len).clone(),
                    "continuation_ids": chunk[prefix_len:].view(1, continuation_len).clone(),
                    "start_offset": start,
                }
            )
        if len(selected) >= num_prompts:
            break

    if len(selected) < num_prompts:
        fallback_starts = [0] if max_start == 0 else torch.linspace(0, max_start, steps=num_prompts).round().long().tolist()
        for start in fallback_starts:
            chunk = all_ids[start:start + total_len]
            if chunk.numel() != total_len:
                continue
            selected.append(
                {
                    "prompt_id": len(selected),
                    "dataset": dataset_name,
                    "input_ids": chunk.view(1, total_len).clone(),
                    "prefix_ids": chunk[:prefix_len].view(1, prefix_len).clone(),
                    "continuation_ids": chunk[prefix_len:].view(1, continuation_len).clone(),
                    "start_offset": start,
                }
            )
            if len(selected) >= num_prompts:
                break

    if selected and len(selected) < num_prompts:
        idx = 0
        while len(selected) < num_prompts:
            base = selected[idx % len(selected)]
            selected.append(
                {
                    "prompt_id": len(selected),
                    "dataset": dataset_name,
                    "input_ids": base["input_ids"].clone(),
                    "prefix_ids": base["prefix_ids"].clone(),
                    "continuation_ids": base["continuation_ids"].clone(),
                    "start_offset": base.get("start_offset"),
                }
            )
            idx += 1

    first_starts = [sample.get("start_offset", 0) for sample in selected[:3]]
    print(f"generated sample count: {len(selected)}")
    print(f"first 3 sample start offsets: {first_starts}")
    return selected


def build_synthetic_eval_samples(tokenizer, num_prompts: int, prefix_len: int, continuation_len: int, seed: int, device=None):
    total_len = prefix_len + continuation_len
    vocab_size = int(getattr(tokenizer, "vocab_size", None) or 50257)
    lower = 100
    upper = min(vocab_size, 50000)
    if upper <= lower:
        upper = max(lower + 1, vocab_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    samples = []
    for prompt_id in range(num_prompts):
        input_ids = torch.randint(low=lower, high=upper, size=(1, total_len), generator=generator, dtype=torch.long)
        samples.append(
            {
                "prompt_id": prompt_id,
                "dataset": "synthetic",
                "input_ids": input_ids,
                "prefix_ids": input_ids[:, :prefix_len].clone(),
                "continuation_ids": input_ids[:, prefix_len:].clone(),
                "start_offset": None,
            }
        )
    print(f"[sample] synthetic sample count={len(samples)} prefix_len={prefix_len} continuation_len={continuation_len} total_len={total_len}")
    if samples:
        print(f"[sample] first synthetic sample token shape={tuple(samples[0]['input_ids'].shape)}")
    return samples


def build_output_path_for_phase23(args):
    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    config = {
        "preset": getattr(args, "preset", None),
        "dataset": args.dataset,
        "strategies": args.strategies,
        "strategy_order": getattr(args, "strategy_order", "default"),
        "hybrid_alpha": getattr(args, "hybrid_alpha", [1.0]),
        "anchor_layers": getattr(args, "anchor_layers", ""),
        "anchor_heads": getattr(args, "anchor_heads", ""),
        "anchor_min_keep_ratio": getattr(args, "anchor_min_keep_ratio", None),
        "budget_ratios": args.budget_ratios,
        "prefix_len": args.prefix_len,
        "continuation_len": args.continuation_len,
        "seed": args.seed,
        "seeds": getattr(args, "seeds", None),
    }
    digest = hashlib.md5(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    filename = f"{DEFAULT_OUTPUT_STEM}__{digest}.csv"
    path = Path(args.output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def make_error_row(args, prompt_id, strategy, budget_ratio, error_msg, effective_budget_ratio=None, status="error", **overrides):
    row = {
        "seed": getattr(args, "seed", ""),
        "dataset": args.dataset,
        "strategy": strategy,
        "budget_ratio": budget_ratio,
        "target_budget_ratio": budget_ratio,
        "effective_budget_ratio": effective_budget_ratio if effective_budget_ratio is not None else budget_ratio,
        "budget_abs_error": "",
        "prompt_id": prompt_id,
        "prefix_len": args.prefix_len,
        "continuation_len": args.continuation_len,
        "full_loss": "",
        "continuation_loss": "",
        "loss_increase": "",
        "full_ppl": "",
        "continuation_ppl": "",
        "logits_mse_to_full": "",
        "top1_agreement_to_full": "",
        "original_state_elements": "",
        "total_cache_elements": "",
        "manageable_state_elements": "",
        "kept_manageable_elements": "",
        "unmanaged_elements": "",
        "kept_state_elements": "",
        "estimated_state_memory_bytes": "",
        "memory_reduction_ratio": "",
        "average_decode_latency_ms": "",
        "tokens_per_second": "",
        "clean_cache_changed": "",
        "alias_check_pass": "",
        "nested_budget_check": "",
        "priority_direction_check": "",
        "full_repeat_check": "",
        "keep_set_hash": "",
        "strategy_order": args.strategy_order if hasattr(args, "strategy_order") else "",
        "hybrid_alpha": "",
        "anchor_min_keep_ratio": "",
        "norm_score": "",
        "sensitivity_score": "",
        "final_priority_score": "",
        "sensitivity_valid": "",
        "sensitivity_nonzero_count": "",
        "sensitivity_max": "",
        "sensitivity_mean_positive": "",
        "sensitivity_source_file": "",
        "sensitivity_score_std": "",
        "final_priority_changed_vs_norm": "",
        "kept_mean_priority": "",
        "dropped_mean_priority": "",
        "priority_top10": "",
        "priority_bottom10": "",
        "status": status,
        "error_msg": error_msg,
    }
    row.update(overrides)
    return row


def make_result_row(
    args,
    sample,
    strategy,
    budget_ratio,
    full_loss,
    continuation_loss,
    full_ppl,
    continuation_ppl,
    logits_mse,
    top1_agreement,
    original_state_elements,
    total_cache_elements,
    manageable_state_elements,
    kept_manageable_elements,
    unmanaged_elements,
    kept_state_elements,
    estimated_state_memory_bytes,
    average_decode_latency_ms,
    tokens_per_second,
    clean_cache_changed,
    alias_check_pass,
    nested_budget_check,
    priority_direction_check,
    full_repeat_check,
    keep_set_hash,
    strategy_order,
    hybrid_alpha,
    anchor_min_keep_ratio,
    norm_score,
    sensitivity_score,
    final_priority_score,
    sensitivity_valid,
    sensitivity_nonzero_count,
    sensitivity_max,
    sensitivity_mean_positive,
    sensitivity_source_file,
    sensitivity_score_std,
    final_priority_changed_vs_norm,
    kept_mean_priority,
    dropped_mean_priority,
    priority_top10,
    priority_bottom10,
    status,
    error_msg,
    effective_budget_ratio=None,
    budget_abs_error=None,
):
    effective_budget_ratio = float(effective_budget_ratio if effective_budget_ratio is not None else budget_ratio)
    budget_abs_error = float(budget_abs_error if budget_abs_error is not None else abs(effective_budget_ratio - budget_ratio))
    return {
        "seed": getattr(args, "seed", ""),
        "dataset": sample["dataset"],
        "strategy": strategy,
        "budget_ratio": float(budget_ratio),
        "target_budget_ratio": float(budget_ratio),
        "effective_budget_ratio": effective_budget_ratio,
        "budget_abs_error": budget_abs_error,
        "prompt_id": int(sample["prompt_id"]),
        "prefix_len": int(args.prefix_len),
        "continuation_len": int(args.continuation_len),
        "full_loss": full_loss,
        "continuation_loss": continuation_loss,
        "loss_increase": (continuation_loss - full_loss) if isinstance(full_loss, (int, float)) and isinstance(continuation_loss, (int, float)) else "",
        "full_ppl": full_ppl,
        "continuation_ppl": continuation_ppl,
        "logits_mse_to_full": logits_mse,
        "top1_agreement_to_full": top1_agreement,
        "original_state_elements": original_state_elements,
        "total_cache_elements": total_cache_elements,
        "manageable_state_elements": manageable_state_elements,
        "kept_manageable_elements": kept_manageable_elements,
        "unmanaged_elements": unmanaged_elements,
        "kept_state_elements": kept_state_elements,
        "estimated_state_memory_bytes": estimated_state_memory_bytes,
        "memory_reduction_ratio": 1.0 - (kept_state_elements / original_state_elements if original_state_elements else 0.0),
        "average_decode_latency_ms": average_decode_latency_ms,
        "tokens_per_second": tokens_per_second,
        "clean_cache_changed": clean_cache_changed,
        "alias_check_pass": alias_check_pass,
        "nested_budget_check": nested_budget_check,
        "priority_direction_check": priority_direction_check,
        "full_repeat_check": full_repeat_check,
        "keep_set_hash": keep_set_hash,
        "strategy_order": strategy_order,
        "hybrid_alpha": hybrid_alpha,
        "anchor_min_keep_ratio": anchor_min_keep_ratio,
        "norm_score": norm_score,
        "sensitivity_score": sensitivity_score,
        "final_priority_score": final_priority_score,
        "sensitivity_valid": sensitivity_valid,
        "sensitivity_nonzero_count": sensitivity_nonzero_count,
        "sensitivity_max": sensitivity_max,
        "sensitivity_mean_positive": sensitivity_mean_positive,
        "sensitivity_source_file": sensitivity_source_file,
        "sensitivity_score_std": sensitivity_score_std,
        "final_priority_changed_vs_norm": final_priority_changed_vs_norm,
        "kept_mean_priority": kept_mean_priority,
        "dropped_mean_priority": dropped_mean_priority,
        "priority_top10": priority_top10,
        "priority_bottom10": priority_bottom10,
        "status": status,
        "error_msg": error_msg,
    }


def clone_packed_cache(packed):
    return clone_cache_deep(packed)


def clone_cache_deep(cache):
    if torch.is_tensor(cache):
        cloned = cache.detach().clone()
        if cloned.data_ptr() == cache.data_ptr():
            raise RuntimeError("clone_cache_deep produced aliased tensor")
        return cloned
    if isinstance(cache, dict):
        return {key: clone_cache_deep(value) for key, value in cache.items()}
    if isinstance(cache, tuple):
        return tuple(clone_cache_deep(value) for value in cache)
    if isinstance(cache, list):
        return [clone_cache_deep(value) for value in cache]
    if dataclasses.is_dataclass(cache):
        kwargs = {field.name: clone_cache_deep(getattr(cache, field.name)) for field in dataclasses.fields(cache)}
        return type(cache)(**kwargs)
    if hasattr(cache, "__dict__") and not isinstance(cache, type):
        cloned = copy.copy(cache)
        for key, value in cache.__dict__.items():
            setattr(cloned, key, clone_cache_deep(value))
        return cloned
    return copy.deepcopy(cache)


def _fingerprint_tensor(tensor):
    tensor_f = tensor.detach().float()
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "sum": float(tensor_f.sum().item()),
        "abs_sum": float(tensor_f.abs().sum().item()),
        "norm": float(tensor_f.norm().item()),
        "data_ptr": int(tensor.data_ptr()),
    }


def _tensor_storage_ptr(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except Exception:
        return int(tensor.storage().data_ptr())


def iter_cache_tensors(cache, prefix="cache"):
    if torch.is_tensor(cache):
        yield prefix, cache
        return
    if isinstance(cache, dict):
        for key, value in cache.items():
            yield from iter_cache_tensors(value, f"{prefix}.{key}")
        return
    if isinstance(cache, tuple):
        for idx, value in enumerate(cache):
            yield from iter_cache_tensors(value, f"{prefix}[{idx}]")
        return
    if isinstance(cache, list):
        for idx, value in enumerate(cache):
            yield from iter_cache_tensors(value, f"{prefix}[{idx}]")
        return
    if dataclasses.is_dataclass(cache):
        for field in dataclasses.fields(cache):
            yield from iter_cache_tensors(getattr(cache, field.name), f"{prefix}.{field.name}")
        return
    if hasattr(cache, "__dict__") and not isinstance(cache, type):
        for key, value in cache.__dict__.items():
            yield from iter_cache_tensors(value, f"{prefix}.{key}")


def cache_fingerprint(cache):
    tensors = []

    def visit(value):
        if torch.is_tensor(value):
            tensors.append(_fingerprint_tensor(value))
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name))
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            for item in value.__dict__.values():
                visit(item)

    visit(cache)
    return tensors


def cache_fingerprint_signature(cache):
    fingerprints = cache_fingerprint(cache)
    total_sum = sum(item["sum"] for item in fingerprints)
    total_abs = sum(item["abs_sum"] for item in fingerprints)
    total_norm = sum(item["norm"] for item in fingerprints)
    total_ptr = sum(item["data_ptr"] for item in fingerprints)
    return {
        "n_tensors": len(fingerprints),
        "total_sum": total_sum,
        "total_abs_sum": total_abs,
        "total_norm": total_norm,
        "total_ptr": total_ptr,
        "fingerprints": fingerprints,
    }


def fingerprint_changed(before, after):
    keys = ["n_tensors", "total_sum", "total_abs_sum", "total_norm"]
    return any(abs(float(before[key]) - float(after[key])) > 1e-8 for key in keys)


def summarize_fingerprint(signature):
    return {
        "n_tensors": signature["n_tensors"],
        "total_sum": round(float(signature["total_sum"]), 6),
        "total_abs_sum": round(float(signature["total_abs_sum"]), 6),
        "total_norm": round(float(signature["total_norm"]), 6),
    }


def check_cache_alias(clean_cache, work_cache, debug: bool = False):
    clean_tensors = list(iter_cache_tensors(clean_cache, "clean"))
    work_tensors = list(iter_cache_tensors(work_cache, "work"))
    alias_pairs = []
    work_by_storage = {}
    for work_path, work_tensor in work_tensors:
        work_by_storage.setdefault(_tensor_storage_ptr(work_tensor), []).append((work_path, work_tensor))
    for clean_path, clean_tensor in clean_tensors:
        storage_ptr = _tensor_storage_ptr(clean_tensor)
        matches = work_by_storage.get(storage_ptr, [])
        for work_path, work_tensor in matches:
            alias_pairs.append(
                {
                    "clean_path": clean_path,
                    "work_path": work_path,
                    "clean_shape": tuple(clean_tensor.shape),
                    "work_shape": tuple(work_tensor.shape),
                    "clean_data_ptr": int(clean_tensor.data_ptr()),
                    "work_data_ptr": int(work_tensor.data_ptr()),
                    "shared_storage_ptr": storage_ptr,
                }
            )
    if debug:
        print("[cache_alias] clean tensor pointers:")
        for path, tensor in clean_tensors:
            print(f"  {path}: data_ptr={int(tensor.data_ptr())} storage_ptr={_tensor_storage_ptr(tensor)} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
        print("[cache_alias] work tensor pointers:")
        for path, tensor in work_tensors:
            print(f"  {path}: data_ptr={int(tensor.data_ptr())} storage_ptr={_tensor_storage_ptr(tensor)} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
        print(f"[cache_alias] shared_storage={bool(alias_pairs)}")
    return len(alias_pairs) == 0, alias_pairs


def build_score_maps(score_info):
    layer_scores = {}
    for row in score_info["layer_rows"]:
        try:
            layer_idx = int(row.get("layer_idx"))
            score = float(row.get("loss_increase", 0.0))
        except Exception:
            continue
        if math.isfinite(score):
            layer_scores[layer_idx] = score
    if not layer_scores:
        layer_scores = {21: 4.0, 20: 3.5, 19: 3.0, 5: 2.8, 14: 2.2, 18: 2.0, 6: 1.8, 17: 1.6}

    head_scores = {}
    for row in score_info["head_rows"]:
        try:
            key = (int(row.get("layer_idx")), int(row.get("head_id")))
            score = float(row.get("loss_increase", 0.0))
        except Exception:
            continue
        if math.isfinite(score):
            head_scores[key] = score
    if not head_scores:
        head_scores = {(21, 15): 4.0, (20, 19): 3.5, (14, 1): 3.0, (20, 16): 2.8, (19, 17): 2.6, (5, 17): 2.4}

    group_scores = {}
    for row in score_info["group_rows"]:
        try:
            key = (int(row.get("layer_idx")), int(row.get("head_id")), int(row.get("channel_group_id")))
            score = float(row.get("loss_increase", 0.0))
        except Exception:
            continue
        if math.isfinite(score):
            group_scores[key] = score

    layer_zscores = build_zscore_map(layer_scores)
    head_zscores = build_zscore_map(head_scores)
    group_zscores = build_zscore_map(group_scores)
    return {
        "layer_scores": layer_scores,
        "head_scores": head_scores,
        "group_scores": group_scores,
        "layer_zscores": layer_zscores,
        "head_zscores": head_zscores,
        "group_zscores": group_zscores,
        "layer_info": score_info.get("layer_info", {}),
        "head_info": score_info.get("head_info", {}),
        "group_info": score_info.get("group_info", {}),
    }


def has_valid_signal(score_map: dict[Any, float]) -> bool:
    if not score_map:
        return False
    values = [float(v) for v in score_map.values() if math.isfinite(float(v))]
    if not values:
        return False
    if max(abs(v) for v in values) <= 0.0:
        return False
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance > 0.0


def build_zscore_map(score_map: dict[Any, float]) -> dict[Any, float]:
    if not has_valid_signal(score_map):
        return {}
    values = [float(v) for v in score_map.values() if math.isfinite(float(v))]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(max(variance, 1e-12))
    return {key: (float(value) - mean) / std for key, value in score_map.items() if math.isfinite(float(value))}


def parse_anchor_layers(spec: str) -> set[int]:
    if not spec.strip():
        return set()
    return {int(token.strip()) for token in spec.split(",") if token.strip()}


def parse_anchor_heads(spec: str) -> set[tuple[int, int]]:
    anchors = set()
    if not spec.strip():
        return anchors
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        layer_str, head_str = token.split(":", 1)
        anchors.add((int(layer_str), int(head_str)))
    return anchors


def expand_strategy_variants(strategies, hybrid_alphas, anchor_boosts):
    expanded = []
    hybrid_set = {"hybrid_norm_layer", "hybrid_norm_head", "hybrid_norm_group", "hybrid_norm_layer_head", "hybrid_norm_layer_head_group"}
    for strategy in strategies:
        if strategy in hybrid_set:
            for alpha in hybrid_alphas:
                expanded.append((strategy, float(alpha), None))
        elif strategy == "anchor_protected_state_norm":
            for anchor_boost in anchor_boosts:
                expanded.append((strategy, None, float(anchor_boost)))
        else:
            expanded.append((strategy, None, None))
    return expanded


def layer_default_score(layer_idx: int):
    high = {21, 20, 19, 5}
    medium = {14, 18, 6, 17}
    if layer_idx in high:
        return 3.0
    if layer_idx in medium:
        return 2.0
    return 1.0


def head_default_score(layer_idx: int, head_id: int):
    top_heads = {(21, 15), (20, 19), (14, 1), (20, 16), (19, 17), (5, 17)}
    return 3.0 if (layer_idx, head_id) in top_heads else 1.0


def group_default_score(layer_idx: int, head_id: int, group_id: int):
    return head_default_score(layer_idx, head_id) + max(0.0, 1.0 - 0.1 * group_id)


def normalize_strategy_order(strategies, strategy_order: str, single_strategy_debug: bool = False):
    ordered = list(strategies)
    if single_strategy_debug:
        non_full = [strategy for strategy in ordered if strategy != "full_state"]
        if len(non_full) != 1:
            raise ValueError("--single_strategy_debug requires exactly one non-full strategy")
        ordered = ["full_state", non_full[0]]
    if strategy_order == "reversed":
        ordered = list(reversed(ordered))
    elif strategy_order == "shuffled":
        rng = random.Random(0)
        ordered = ordered[:]
        rng.shuffle(ordered)
    return ordered


DETERMINISTIC_NESTED_STRATEGIES = {
    "state_norm_keep",
    "layer_aware_statecache",
    "head_aware_statecache",
    "group_aware_statecache",
    "random_drop",
    "hybrid_norm_layer",
    "hybrid_norm_head",
    "hybrid_norm_group",
    "hybrid_norm_layer_head",
    "hybrid_norm_layer_head_group",
    "anchor_protected_state_norm",
}


def _unit_identity(unit):
    return (
        int(unit["layer_idx"]),
        int(unit["head_id"]),
        int(unit["channel_group_id"]),
        int(unit["ssm_channel_index"]),
    )


def _stable_int_hash(payload) -> int:
    return int(hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:8], 16)


def _compute_unit_state_norm(unit, state_norm_maps):
    layer_norms, head_norms, group_norms = state_norm_maps
    layer_idx = unit["layer_idx"]
    head_id = unit["head_id"]
    group_id = unit["channel_group_id"]
    layer_norm = float(layer_norms.get(layer_idx, 0.0))
    head_norm = float(head_norms.get((layer_idx, head_id), layer_norm))
    group_norm = float(group_norms.get((layer_idx, head_id, group_id), head_norm))
    return 0.5 * head_norm + 0.5 * group_norm


def build_norm_zscores(units, state_norm_maps):
    raw = [_compute_unit_state_norm(unit, state_norm_maps) for unit in units]
    if not raw:
        return {}, {}
    mean = sum(raw) / len(raw)
    variance = sum((value - mean) ** 2 for value in raw) / len(raw)
    std = math.sqrt(max(variance, 1e-12))
    norm_scores = {}
    norm_zscores = {}
    for unit, value in zip(units, raw):
        key = _unit_identity(unit)
        norm_scores[key] = float(value)
        norm_zscores[key] = (float(value) - mean) / std
    return norm_scores, norm_zscores


def _sensitivity_info_for_strategy(strategy, score_maps):
    if strategy in {"hybrid_norm_layer", "layer_aware_statecache"}:
        return score_maps["layer_info"]
    if strategy in {"hybrid_norm_head", "head_aware_statecache"}:
        return score_maps["head_info"]
    if strategy in {"hybrid_norm_group", "group_aware_statecache", "hybrid_norm_layer_head_group"}:
        return score_maps["group_info"]
    if strategy == "hybrid_norm_layer_head":
        if score_maps["layer_info"].get("valid"):
            return score_maps["layer_info"]
        return score_maps["head_info"]
    return {"valid": True, "path": "", "nonzero_count": 0, "max_loss_increase": 0.0, "mean_positive_loss_increase": 0.0}


def build_priority_lists_for_units(
    units,
    strategy,
    score_maps,
    state_norm_maps,
    rng,
    prompt_seed: int,
    hybrid_alpha: float | None = None,
    anchor_layers: set[int] | None = None,
    anchor_heads: set[tuple[int, int]] | None = None,
    anchor_boost: float = 0.0,
):
    layer_scores = score_maps["layer_scores"]
    head_scores = score_maps["head_scores"]
    group_scores = score_maps["group_scores"]
    layer_zscores = score_maps["layer_zscores"]
    head_zscores = score_maps["head_zscores"]
    group_zscores = score_maps["group_zscores"]
    priority_units = []
    local_rng = random.Random(prompt_seed * 100003 + _stable_int_hash([strategy, hybrid_alpha]))
    norm_scores, norm_zscores = build_norm_zscores(units, state_norm_maps)
    alpha = float(hybrid_alpha if hybrid_alpha is not None else 1.0)
    anchor_layers = anchor_layers or set()
    anchor_heads = anchor_heads or set()
    sensitivity_info = _sensitivity_info_for_strategy(strategy, score_maps)
    final_scores = []
    for idx, unit in enumerate(units):
        layer_idx = unit["layer_idx"]
        head_id = unit["head_id"]
        group_id = unit["channel_group_id"]
        stable_key = _unit_identity(unit)
        unit_state_norm = float(norm_scores.get(stable_key, 0.0))
        norm_z = float(norm_zscores.get(stable_key, 0.0))
        layer_sensitivity = float(layer_zscores.get(layer_idx, 0.0))
        head_sensitivity = float(head_zscores.get((layer_idx, head_id), 0.0))
        group_sensitivity = float(group_zscores.get((layer_idx, head_id, group_id), 0.0))
        sensitivity_score = 0.0
        stable_key = _unit_identity(unit)
        if strategy == "random_drop":
            random_seed = _stable_int_hash([prompt_seed, strategy, list(stable_key)])
            score = random.Random(random_seed).random()
        elif strategy == "uniform_drop":
            score = local_rng.random()
        elif strategy == "state_norm_keep":
            score = unit_state_norm
        elif strategy == "layer_aware_statecache":
            sensitivity_score = float(layer_scores.get(layer_idx, layer_default_score(layer_idx)))
            score = sensitivity_score + 1e-6 * unit_state_norm
        elif strategy == "head_aware_statecache":
            sensitivity_score = float(head_scores.get((layer_idx, head_id), head_default_score(layer_idx, head_id)))
            score = sensitivity_score + 1e-6 * unit_state_norm
        elif strategy == "group_aware_statecache":
            sensitivity_score = float(group_scores.get((layer_idx, head_id, group_id), group_default_score(layer_idx, head_id, group_id)))
            score = sensitivity_score + 1e-6 * unit_state_norm
        elif strategy == "hybrid_norm_layer":
            sensitivity_score = layer_sensitivity
            score = norm_z + alpha * layer_sensitivity
        elif strategy == "hybrid_norm_head":
            sensitivity_score = head_sensitivity
            score = norm_z + alpha * head_sensitivity
        elif strategy == "hybrid_norm_group":
            sensitivity_score = group_sensitivity
            score = norm_z + alpha * group_sensitivity
        elif strategy == "hybrid_norm_layer_head":
            sensitivity_score = 0.5 * (layer_sensitivity + head_sensitivity)
            score = norm_z + alpha * sensitivity_score
        elif strategy == "hybrid_norm_layer_head_group":
            sensitivity_score = (layer_sensitivity + head_sensitivity + group_sensitivity) / 3.0
            score = norm_z + alpha * sensitivity_score
        elif strategy == "anchor_protected_state_norm":
            anchor_indicator = 1.0 if (layer_idx in anchor_layers or (layer_idx, head_id) in anchor_heads) else 0.0
            sensitivity_score = anchor_indicator
            score = norm_z + anchor_boost * anchor_indicator
        else:
            raise ValueError(f"Unsupported strategy for scoring: {strategy}")
        enriched = dict(unit)
        enriched["state_norm"] = unit_state_norm
        enriched["norm_score"] = norm_z if strategy.startswith("hybrid_") else unit_state_norm
        enriched["sensitivity_score"] = sensitivity_score
        enriched["final_priority_score"] = score
        enriched["priority_score"] = score
        priority_units.append(enriched)
        final_scores.append(score)
    priority_units.sort(key=lambda item: (item["priority_score"],) + _unit_identity(item), reverse=True)
    if final_scores:
        mean = sum(final_scores) / len(final_scores)
        variance = sum((value - mean) ** 2 for value in final_scores) / len(final_scores)
        score_std = math.sqrt(max(variance, 1e-12))
    else:
        score_std = 0.0
    return priority_units, {
        "sensitivity_valid": bool(sensitivity_info.get("valid", False)),
        "sensitivity_nonzero_count": int(sensitivity_info.get("nonzero_count", 0)),
        "sensitivity_max": float(sensitivity_info.get("max_loss_increase", 0.0)),
        "sensitivity_mean_positive": float(sensitivity_info.get("mean_positive_loss_increase", 0.0)),
        "sensitivity_source_file": str(sensitivity_info.get("path", "")),
        "sensitivity_score_std": float(score_std),
    }


def get_keep_signature(units):
    payload = [
        (int(unit["layer_idx"]), int(unit["head_id"]), int(unit["channel_group_id"]), int(unit["ssm_channel_index"]))
        for unit in units
    ]
    return hashlib.md5(json.dumps(payload).encode("utf-8")).hexdigest()


def collect_total_elements(cache_entry):
    total_elements = 0
    total_bytes = 0
    for value in cache_entry:
        if torch.is_tensor(value):
            total_elements += int(value.numel())
            total_bytes += int(value.numel() * value.element_size())
    return total_elements, total_bytes


def collect_cache_inventory(packed_cache, mixer_pairs):
    total_cache_elements = 0
    total_cache_bytes = 0
    manageable_units = []
    manageable_state_elements = 0
    manageable_state_bytes = 0
    warnings = []

    for layer_idx, cache_entry in packed_cache["key_value_memory_dict"].items():
        if isinstance(cache_entry, tuple):
            total_elems, total_bytes = collect_total_elements(cache_entry)
            total_cache_elements += total_elems
            total_cache_bytes += total_bytes
        elif isinstance(cache_entry, dict):
            for value in cache_entry.values():
                if torch.is_tensor(value):
                    total_cache_elements += int(value.numel())
                    total_cache_bytes += int(value.numel() * value.element_size())
        elif torch.is_tensor(cache_entry):
            total_cache_elements += int(cache_entry.numel())
            total_cache_bytes += int(cache_entry.numel() * cache_entry.element_size())

    mixer_map = {layer_idx: mixer for layer_idx, mixer in mixer_pairs}
    for layer_idx, mixer in mixer_pairs:
        if layer_idx not in packed_cache["key_value_memory_dict"]:
            continue
        cache_entry = packed_cache["key_value_memory_dict"][layer_idx]
        if not (isinstance(cache_entry, tuple) and len(cache_entry) >= 2 and torch.is_tensor(cache_entry[0]) and torch.is_tensor(cache_entry[1])):
            warnings.append(f"layer {layer_idx}: unsupported cache entry type {type(cache_entry)}")
            continue
        conv_state, ssm_state = cache_entry[:2]
        d_inner = int(getattr(mixer, "d_inner", 0))
        nheads = int(getattr(mixer, "nheads", 0))
        headdim = int(getattr(mixer, "headdim", 0))
        if d_inner <= 0 or nheads <= 0 or headdim <= 0:
            warnings.append(f"layer {layer_idx}: missing mixer dimensions")
            continue
        if conv_state.ndim < 3 or ssm_state.ndim < 4:
            warnings.append(f"layer {layer_idx}: unsupported tensor shapes conv={tuple(conv_state.shape)} ssm={tuple(ssm_state.shape)}")
            continue
        manageable_conv_dim = min(d_inner, int(conv_state.shape[1]))
        manageable_ssm_heads = min(nheads, int(ssm_state.shape[1]))
        manageable_ssm_dim = min(headdim, int(ssm_state.shape[2]))
        if manageable_conv_dim <= 0 or manageable_ssm_heads <= 0 or manageable_ssm_dim <= 0:
            warnings.append(f"layer {layer_idx}: non-positive manageable dims")
            continue
        for head_id in range(manageable_ssm_heads):
            for channel_offset in range(manageable_ssm_dim):
                conv_idx = head_id * headdim + channel_offset
                if conv_idx >= manageable_conv_dim:
                    continue
                conv_slice = conv_state[:, conv_idx, :]
                ssm_slice = ssm_state[:, head_id, channel_offset, :]
                if not (torch.is_tensor(conv_slice) and torch.is_tensor(ssm_slice)):
                    continue
                unit_elements = int(conv_slice.numel() + ssm_slice.numel())
                unit_bytes = int(conv_slice.numel() * conv_slice.element_size() + ssm_slice.numel() * ssm_slice.element_size())
                unit = {
                    "layer_idx": layer_idx,
                    "head_id": head_id,
                    "channel_group_id": channel_offset // GROUP_SIZE,
                    "tensor_name": "conv_x+ssm_state",
                    "conv_index": conv_idx,
                    "ssm_channel_index": channel_offset,
                    "conv_slice": (conv_idx, slice(None)),
                    "ssm_slice": (head_id, channel_offset, slice(None)),
                    "num_elements": unit_elements,
                    "num_bytes": unit_bytes,
                    "state_norm": None,
                    "priority_score": 0.0,
                }
                manageable_units.append(unit)
                manageable_state_elements += unit_elements
                manageable_state_bytes += unit_bytes

    unmanaged_elements = max(0, total_cache_elements - manageable_state_elements)
    unmanaged_bytes = max(0, total_cache_bytes - manageable_state_bytes)
    return {
        "manageable_units": manageable_units,
        "total_cache_elements": total_cache_elements,
        "total_cache_bytes": total_cache_bytes,
        "manageable_state_elements": manageable_state_elements,
        "manageable_state_bytes": manageable_state_bytes,
        "unmanaged_elements": unmanaged_elements,
        "unmanaged_bytes": unmanaged_bytes,
        "warnings": warnings,
        "mixer_map": mixer_map,
    }


def assign_unit_scores(units, strategy, packed_cache, mixer_pairs, score_maps, state_norm_maps, rng):
    layer_scores, head_scores, group_scores = score_maps
    layer_norms, head_norms, group_norms = state_norm_maps
    scored_units = []
    for unit in units:
        layer_idx = unit["layer_idx"]
        head_id = unit["head_id"]
        group_id = unit["channel_group_id"]
        layer_norm = float(layer_norms.get(layer_idx, 0.0))
        head_norm = float(head_norms.get((layer_idx, head_id), layer_norm))
        group_norm = float(group_norms.get((layer_idx, head_id, group_id), head_norm))
        unit_state_norm = 0.5 * head_norm + 0.5 * group_norm
        if strategy == "random_drop":
            score = rng.random()
        elif strategy == "uniform_drop":
            score = rng.random()
        elif strategy == "state_norm_keep":
            score = unit_state_norm
        elif strategy == "layer_aware_statecache":
            score = float(layer_scores.get(layer_idx, layer_default_score(layer_idx))) + 1e-6 * unit_state_norm
        elif strategy == "head_aware_statecache":
            score = float(head_scores.get((layer_idx, head_id), head_default_score(layer_idx, head_id))) + 1e-6 * unit_state_norm
        elif strategy == "group_aware_statecache":
            score = float(group_scores.get((layer_idx, head_id, group_id), group_default_score(layer_idx, head_id, group_id))) + 1e-6 * unit_state_norm
        else:
            raise ValueError(f"Unsupported strategy for scoring: {strategy}")
        scored_unit = dict(unit)
        scored_unit["state_norm"] = unit_state_norm
        scored_unit["priority_score"] = score
        scored_units.append(scored_unit)
    scored_units.sort(key=lambda item: item["priority_score"], reverse=True)
    return scored_units


def select_units_exact_budget(scored_units, target_budget_ratio):
    total_elements = sum(int(unit["num_elements"]) for unit in scored_units)
    target_keep_elements = int(round(target_budget_ratio * total_elements))
    selected = []
    kept_elements = 0
    for unit in scored_units:
        if kept_elements >= target_keep_elements and selected:
            break
        selected.append(unit)
        kept_elements += int(unit["num_elements"])
    if selected and kept_elements > target_keep_elements:
        best_without_last = selected[:-1]
        best_without_last_elements = kept_elements - int(selected[-1]["num_elements"])
        if abs(best_without_last_elements - target_keep_elements) <= abs(kept_elements - target_keep_elements):
            selected = best_without_last
            kept_elements = best_without_last_elements
    if selected and kept_elements < target_keep_elements:
        # add one more unit if we are still short
        for unit in scored_units[len(selected):]:
            selected.append(unit)
            kept_elements += int(unit["num_elements"])
            if kept_elements >= target_keep_elements:
                break
    effective_budget_ratio = kept_elements / total_elements if total_elements else 0.0
    budget_abs_error = abs(effective_budget_ratio - target_budget_ratio)
    return selected, kept_elements, target_keep_elements, effective_budget_ratio, budget_abs_error, total_elements


def select_units_with_anchor_protection(scored_units, target_budget_ratio, anchor_layers, anchor_heads, anchor_min_keep_ratio):
    total_elements = sum(int(unit["num_elements"]) for unit in scored_units)
    target_keep_elements = int(round(target_budget_ratio * total_elements))
    anchor_units = [
        unit for unit in scored_units
        if int(unit["layer_idx"]) in anchor_layers or (int(unit["layer_idx"]), int(unit["head_id"])) in anchor_heads
    ]
    non_anchor_units = [
        unit for unit in scored_units
        if unit not in anchor_units
    ]
    anchor_total = sum(int(unit["num_elements"]) for unit in anchor_units)
    required_anchor_keep = int(round(anchor_min_keep_ratio * anchor_total))
    selected = []
    kept_elements = 0

    for unit in anchor_units:
        if kept_elements >= required_anchor_keep and selected:
            break
        selected.append(unit)
        kept_elements += int(unit["num_elements"])

    if kept_elements > target_keep_elements:
        print(
            f"[WARN] anchor_protected_state_norm target budget too low; prioritizing anchor keep "
            f"required_anchor_keep={required_anchor_keep} target_keep_elements={target_keep_elements}"
        )
        target_keep_elements = kept_elements

    if kept_elements < target_keep_elements:
        for unit in non_anchor_units:
            selected.append(unit)
            kept_elements += int(unit["num_elements"])
            if kept_elements >= target_keep_elements:
                break

    if kept_elements < target_keep_elements:
        for unit in anchor_units[len(selected):]:
            selected.append(unit)
            kept_elements += int(unit["num_elements"])
            if kept_elements >= target_keep_elements:
                break

    effective_budget_ratio = kept_elements / total_elements if total_elements else 0.0
    budget_abs_error = abs(effective_budget_ratio - target_budget_ratio)
    return selected, kept_elements, target_keep_elements, effective_budget_ratio, budget_abs_error, total_elements


def apply_unit_selection_to_cache(cache_copy, dropped_units):
    for unit in dropped_units:
        layer_idx = unit["layer_idx"]
        if layer_idx not in cache_copy["key_value_memory_dict"]:
            continue
        conv_state, ssm_state = cache_copy["key_value_memory_dict"][layer_idx]
        conv_idx = int(unit["conv_index"])
        head_id = int(unit["head_id"])
        channel_offset = int(unit["ssm_channel_index"])
        if conv_state.ndim >= 3 and conv_idx < conv_state.shape[1]:
            conv_state[:, conv_idx, :] = 0
        if ssm_state.ndim >= 4 and head_id < ssm_state.shape[1] and channel_offset < ssm_state.shape[2]:
            ssm_state[:, head_id, channel_offset, :] = 0


def format_top_units(scored_units, top_k=10):
    formatted = []
    for unit in scored_units[:top_k]:
        formatted.append(
            {
                "layer": unit["layer_idx"],
                "head": unit["head_id"],
                "group": unit["channel_group_id"],
                "score": round(float(unit["priority_score"]), 6),
                "num_elements": int(unit["num_elements"]),
            }
        )
    return formatted


def compute_state_norm_scores(packed_cache, mixer_pairs):
    return extract_state_norms(packed_cache, mixer_pairs)


def validate_priority_direction(strategy, kept_units, dropped_units):
    if strategy not in {"state_norm_keep", "hybrid_norm_layer", "hybrid_norm_head", "hybrid_norm_layer_head", "anchor_protected_state_norm"}:
        return True, ""
    kept_norms = [float(unit.get("state_norm", 0.0)) for unit in kept_units]
    dropped_norms = [float(unit.get("state_norm", 0.0)) for unit in dropped_units]
    kept_mean = sum(kept_norms) / len(kept_norms) if kept_norms else 0.0
    dropped_mean = sum(dropped_norms) / len(dropped_norms) if dropped_norms else 0.0
    return kept_mean > dropped_mean, f"kept_mean_norm={kept_mean:.6f} dropped_mean_norm={dropped_mean:.6f}"


def validate_nested_keep_sets(strategy, keep_sets_by_budget, budget_ratios):
    if strategy not in DETERMINISTIC_NESTED_STRATEGIES or len(budget_ratios) <= 1:
        return True, ""
    ordered = sorted(float(x) for x in budget_ratios)
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        lower_set = keep_sets_by_budget.get(lower, set())
        upper_set = keep_sets_by_budget.get(upper, set())
        if not lower_set.issubset(upper_set):
            return False, f"keep_set@{lower} not subset of keep_set@{upper}"
    return True, ""


def run_full_state_from_clean(model, clean_cache, prefill_logits, continuation_ids, device):
    work_cache = clone_cache_deep(clean_cache)
    logits, latency_ms, tokens = decode_continuation(model, work_cache, prefill_logits, continuation_ids, device)
    return logits, latency_ms, tokens


def exact_budget_allocate(
    strategy,
    budget_ratio,
    packed_cache,
    mixer_pairs,
    score_maps,
    debug_budget: bool = False,
    seed: int = 0,
    hybrid_alpha: float | None = None,
    anchor_layers: set[int] | None = None,
    anchor_heads: set[tuple[int, int]] | None = None,
    anchor_min_keep_ratio: float = 0.95,
    anchor_mode: str = "soft",
    anchor_boost: float = 0.0,
):
    cache_copy = clone_cache_deep(packed_cache)
    inventory = collect_cache_inventory(cache_copy, mixer_pairs)
    if inventory["warnings"]:
        for warning in inventory["warnings"]:
            print(f"[WARN] {warning}")

    manageable_units = inventory["manageable_units"]
    total_cache_elements = inventory["total_cache_elements"]
    total_cache_bytes = inventory["total_cache_bytes"]
    manageable_state_elements = inventory["manageable_state_elements"]
    manageable_state_bytes = inventory["manageable_state_bytes"]
    unmanaged_elements = inventory["unmanaged_elements"]
    unmanaged_bytes = inventory["unmanaged_bytes"]

    if strategy == "full_state":
        stats = {
            "total_cache_elements": total_cache_elements,
            "manageable_state_elements": manageable_state_elements,
            "kept_manageable_elements": manageable_state_elements,
            "unmanaged_elements": unmanaged_elements,
            "target_budget_ratio": 1.0,
            "effective_budget_ratio": 1.0 if manageable_state_elements else 0.0,
            "budget_abs_error": 0.0,
            "original_state_elements": total_cache_elements,
            "kept_state_elements": total_cache_elements,
            "estimated_state_memory_bytes": total_cache_bytes,
            "memory_reduction_ratio": 0.0,
            "status": "functional_pass",
            "error_msg": "",
            "selected_units": [],
            "target_keep_elements": manageable_state_elements,
            "priority_units": [],
        }
        return cache_copy, stats

    if manageable_state_elements <= 0 or not manageable_units:
        stats = {
            "total_cache_elements": total_cache_elements,
            "manageable_state_elements": 0,
            "kept_manageable_elements": 0,
            "unmanaged_elements": total_cache_elements,
            "target_budget_ratio": budget_ratio,
            "effective_budget_ratio": 0.0,
            "budget_abs_error": abs(budget_ratio),
            "original_state_elements": total_cache_elements,
            "kept_state_elements": unmanaged_elements,
            "estimated_state_memory_bytes": total_cache_bytes,
            "memory_reduction_ratio": 0.0,
            "status": "error",
            "error_msg": "no manageable state units",
            "selected_units": [],
            "target_keep_elements": 0,
            "priority_units": [],
        }
        return cache_copy, stats

    state_norm_maps = compute_state_norm_scores(packed_cache, mixer_pairs)
    rng = random.Random(seed)
    scored_units, sensitivity_stats = build_priority_lists_for_units(
        manageable_units,
        strategy,
        score_maps,
        state_norm_maps,
        rng,
        seed,
        hybrid_alpha=hybrid_alpha,
        anchor_layers=anchor_layers,
        anchor_heads=anchor_heads,
        anchor_boost=anchor_boost,
    )
    if strategy == "anchor_protected_state_norm" and anchor_mode == "hard":
        selected_units, kept_manageable_elements, target_keep_elements, effective_budget_ratio, budget_abs_error, _ = select_units_with_anchor_protection(
            scored_units,
            budget_ratio,
            anchor_layers or set(),
            anchor_heads or set(),
            anchor_min_keep_ratio,
        )
    else:
        selected_units, kept_manageable_elements, target_keep_elements, effective_budget_ratio, budget_abs_error, _ = select_units_exact_budget(
            scored_units,
            budget_ratio,
        )
    selected_ids = {_unit_identity(unit) for unit in selected_units}
    dropped_units = [unit for unit in scored_units if _unit_identity(unit) not in selected_ids]
    apply_unit_selection_to_cache(cache_copy, dropped_units)
    selected_bytes = sum(int(unit["num_bytes"]) for unit in selected_units)
    estimated_state_memory_bytes = unmanaged_bytes + selected_bytes
    kept_state_elements = unmanaged_elements + kept_manageable_elements
    if debug_budget and strategy == "state_norm_keep":
        kept_norms = [float(unit.get("state_norm", 0.0)) for unit in selected_units]
        dropped_norms = [float(unit.get("state_norm", 0.0)) for unit in dropped_units]
        kept_mean_norm = sum(kept_norms) / len(kept_norms) if kept_norms else 0.0
        dropped_mean_norm = sum(dropped_norms) / len(dropped_norms) if dropped_norms else 0.0
        print(f"[budget] state_norm_keep kept_mean_norm={kept_mean_norm:.6f} dropped_mean_norm={dropped_mean_norm:.6f}")
        print(f"[budget] state_norm_keep kept_top10={[(u['layer_idx'], u['head_id'], u['channel_group_id'], round(float(u.get('state_norm', 0.0)), 6)) for u in selected_units[:10]]}")
        print(f"[budget] state_norm_keep dropped_bottom10={[(u['layer_idx'], u['head_id'], u['channel_group_id'], round(float(u.get('state_norm', 0.0)), 6)) for u in dropped_units[-10:]]}")
    stats = {
        "total_cache_elements": total_cache_elements,
        "manageable_state_elements": manageable_state_elements,
        "kept_manageable_elements": kept_manageable_elements,
        "unmanaged_elements": unmanaged_elements,
        "target_budget_ratio": budget_ratio,
        "effective_budget_ratio": effective_budget_ratio,
        "budget_abs_error": budget_abs_error,
        "original_state_elements": total_cache_elements,
        "kept_state_elements": kept_state_elements,
        "estimated_state_memory_bytes": estimated_state_memory_bytes,
        "memory_reduction_ratio": 1.0 - (kept_state_elements / total_cache_elements if total_cache_elements else 0.0),
        "status": "ok" if budget_abs_error <= 0.02 else "budget_mismatch",
        "error_msg": "",
        "selected_units": selected_units,
        "dropped_units": dropped_units,
        "target_keep_elements": target_keep_elements,
        "priority_units": format_top_units(scored_units),
        "priority_top10": format_top_units(scored_units, 10),
        "priority_bottom10": format_top_units(list(reversed(scored_units)), 10),
        "keep_set_hash": get_keep_signature(selected_units),
        "hybrid_alpha": hybrid_alpha,
        "anchor_min_keep_ratio": anchor_min_keep_ratio if strategy == "anchor_protected_state_norm" else "",
        "norm_score": (sum(float(unit.get("norm_score", 0.0)) for unit in selected_units) / len(selected_units)) if selected_units else 0.0,
        "sensitivity_score": (sum(float(unit.get("sensitivity_score", 0.0)) for unit in selected_units) / len(selected_units)) if selected_units else 0.0,
        "final_priority_score": (sum(float(unit.get("final_priority_score", unit.get("priority_score", 0.0))) for unit in selected_units) / len(selected_units)) if selected_units else 0.0,
        "sensitivity_valid": sensitivity_stats.get("sensitivity_valid", False),
        "sensitivity_nonzero_count": sensitivity_stats.get("sensitivity_nonzero_count", 0),
        "sensitivity_max": sensitivity_stats.get("sensitivity_max", 0.0),
        "sensitivity_mean_positive": sensitivity_stats.get("sensitivity_mean_positive", 0.0),
        "sensitivity_source_file": sensitivity_stats.get("sensitivity_source_file", ""),
        "sensitivity_score_std": sensitivity_stats.get("sensitivity_score_std", 0.0),
        "kept_mean_priority": (sum(float(unit["priority_score"]) for unit in selected_units) / len(selected_units)) if selected_units else 0.0,
        "dropped_mean_priority": (sum(float(unit["priority_score"]) for unit in scored_units[len(selected_units):]) / max(1, len(scored_units[len(selected_units):]))) if len(scored_units) > len(selected_units) else 0.0,
    }
    if debug_budget and strategy != "full_state":
        print(f"[budget] strategy={strategy}")
        print(f"[budget] requested budget_ratio={budget_ratio}")
        print(f"[budget] manageable_state_elements={manageable_state_elements}")
        print(f"[budget] target_keep_elements={target_keep_elements}")
        print(f"[budget] kept_manageable_elements={kept_manageable_elements}")
        print(f"[budget] effective_budget_ratio={effective_budget_ratio:.6f}")
        print(f"[budget] budget_abs_error={budget_abs_error:.6f}")
        print(f"[budget] top 10 priority units={format_top_units(scored_units)}")
    if budget_abs_error > 0.02:
        print(f"[WARN] budget mismatch for strategy={strategy}: requested={budget_ratio} effective={effective_budget_ratio:.6f} abs_error={budget_abs_error:.6f}")
    return cache_copy, stats


def build_synthetic_logits_targets(input_ids, prefix_len, continuation_len):
    prefix_ids = input_ids[:, :prefix_len]
    continuation_ids = input_ids[:, prefix_len:prefix_len + continuation_len]
    return prefix_ids, continuation_ids


@torch.no_grad()
def prefill_prefix(model, prefix_ids: torch.Tensor, max_seqlen: int, device: str, cache_device: str):
    inference_params = build_inference_params(max_seqlen=max_seqlen, batch_size=1)
    batch = prefix_ids.to(device)
    synchronize_if_cuda(device)
    outputs = model(batch, inference_params=inference_params, num_last_tokens=1)
    synchronize_if_cuda(device)
    inference_params.seqlen_offset = prefix_ids.shape[1]
    return outputs.logits.float().cpu(), pack_inference_params(inference_params, cache_device=cache_device)


@torch.no_grad()
def decode_continuation(model, packed_cache, prefill_last_logits, continuation_ids: torch.Tensor, device: str):
    inference_params = restore_inference_params(packed_cache, device)
    logits_chunks = []
    start_time = time.perf_counter()
    token_count = continuation_ids.shape[1]
    if token_count <= 0:
        return torch.empty(1, 0, 0), 0.0, 0
    logits_chunks.append(prefill_last_logits)
    for token in continuation_ids[0, :-1]:
        token_batch = token.view(1, 1).to(device)
        synchronize_if_cuda(device)
        outputs = model(token_batch, inference_params=inference_params, num_last_tokens=1)
        synchronize_if_cuda(device)
        logits_chunks.append(outputs.logits.float().cpu())
        inference_params.seqlen_offset += 1
    total_latency_ms = (time.perf_counter() - start_time) * 1000.0
    logits = torch.cat(logits_chunks, dim=1) if logits_chunks else torch.empty(1, 0, 0)
    return logits, total_latency_ms, token_count


def compute_logits_metrics(full_logits, candidate_logits):
    if full_logits.shape != candidate_logits.shape:
        raise ValueError(f"Logits shape mismatch: full={tuple(full_logits.shape)} candidate={tuple(candidate_logits.shape)}")
    mse = float(F.mse_loss(candidate_logits, full_logits).item())
    full_top1 = full_logits.argmax(dim=-1)
    cand_top1 = candidate_logits.argmax(dim=-1)
    top1_agreement = float((full_top1 == cand_top1).float().mean().item())
    max_abs = float((candidate_logits - full_logits).abs().max().item())
    return mse, top1_agreement, max_abs


def compute_loss_and_ppl(logits, targets):
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean").item()
    ppl = math.exp(loss)
    return loss, ppl


def write_rows(output_path: Path, rows: list[dict[str, Any]], append: bool):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and output_path.exists() else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def safe_float(value):
    try:
        if value is None or value == "":
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    except Exception:
        return None


def safe_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def strategy_display_name(strategy: str, hybrid_alpha: Any):
    if hybrid_alpha in (None, "", "None"):
        return strategy
    return f"{strategy}[alpha={hybrid_alpha}]"


def mean_std(values: list[float]):
    if not values:
        return "", ""
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def build_phase23_summary(output_csv_path: Path, summary_output_path: Path = SUMMARY_OUTPUT_PATH):
    if not output_csv_path.exists():
        print(f"[WARN] output csv missing, skip summary: {output_csv_path}")
        return []

    with output_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)

    eval_rows = []
    for row in raw_rows:
        if row.get("strategy") == "__prompt__":
            continue
        loss_increase = safe_float(row.get("loss_increase"))
        if loss_increase is None:
            continue
        row["_loss_increase"] = loss_increase
        row["_continuation_ppl"] = safe_float(row.get("continuation_ppl"))
        row["_top1_agreement"] = safe_float(row.get("top1_agreement_to_full"))
        row["_effective_budget_ratio"] = safe_float(row.get("effective_budget_ratio"))
        row["_memory_reduction_ratio"] = safe_float(row.get("memory_reduction_ratio"))
        row["_budget_ratio"] = safe_float(row.get("budget_ratio"))
        row["_seed"] = safe_int(row.get("seed"))
        row["_prompt_id"] = safe_int(row.get("prompt_id"))
        eval_rows.append(row)

    if not eval_rows:
        print(f"[WARN] no valid eval rows in {output_csv_path}")
        return []

    grouped = defaultdict(list)
    for row in eval_rows:
        key = (row.get("strategy", ""), row.get("hybrid_alpha", ""), row.get("budget_ratio", ""))
        grouped[key].append(row)

    summary_rows = []
    by_budget = defaultdict(list)
    for key, rows in grouped.items():
        strategy, hybrid_alpha, budget_ratio = key
        loss_vals = [row["_loss_increase"] for row in rows if row["_loss_increase"] is not None]
        ppl_vals = [row["_continuation_ppl"] for row in rows if row["_continuation_ppl"] is not None]
        top1_vals = [row["_top1_agreement"] for row in rows if row["_top1_agreement"] is not None]
        eff_vals = [row["_effective_budget_ratio"] for row in rows if row["_effective_budget_ratio"] is not None]
        mem_vals = [row["_memory_reduction_ratio"] for row in rows if row["_memory_reduction_ratio"] is not None]
        mean_loss, std_loss = mean_std(loss_vals)
        mean_ppl, std_ppl = mean_std(ppl_vals)
        mean_top1, std_top1 = mean_std(top1_vals)
        mean_eff, _ = mean_std(eff_vals)
        mean_mem, _ = mean_std(mem_vals)
        prompt_pairs = {(row.get("seed", ""), row.get("prompt_id", "")) for row in rows}
        seeds = {row.get("seed", "") for row in rows if row.get("seed", "") != ""}
        summary = {
            "strategy": strategy,
            "hybrid_alpha": hybrid_alpha,
            "budget_ratio": safe_float(budget_ratio),
            "mean_loss_increase": mean_loss,
            "std_loss_increase": std_loss,
            "mean_continuation_ppl": mean_ppl,
            "std_continuation_ppl": std_ppl,
            "mean_top1_agreement": mean_top1,
            "std_top1_agreement": std_top1,
            "mean_effective_budget_ratio": mean_eff,
            "mean_memory_reduction_ratio": mean_mem,
            "num_prompts": len(prompt_pairs),
            "num_seeds": len(seeds),
            "rank_by_loss": "",
            "rank_by_top1": "",
            "best_by_loss": False,
            "best_by_top1": False,
            "best_by_pareto": False,
            "pareto_score": (mean_loss + 0.1 * mean_mem) if mean_loss is not None and mean_mem is not None else "",
        }
        summary_rows.append(summary)
        by_budget[summary["budget_ratio"]].append(summary)

    best_table = []
    for budget_ratio, rows in by_budget.items():
        rows.sort(key=lambda item: (float("inf") if item["mean_loss_increase"] is None else item["mean_loss_increase"], strategy_display_name(item["strategy"], item["hybrid_alpha"])))
        for rank, row in enumerate(rows, start=1):
            row["rank_by_loss"] = rank
        rows_top1 = sorted(rows, key=lambda item: (float("inf") if item["mean_top1_agreement"] is None else -item["mean_top1_agreement"], strategy_display_name(item["strategy"], item["hybrid_alpha"])))
        for rank, row in enumerate(rows_top1, start=1):
            row["rank_by_top1"] = rank
        best_loss = rows[0] if rows else None
        best_top1 = rows_top1[0] if rows_top1 else None
        rows_pareto = sorted(rows, key=lambda item: (float("inf") if item["pareto_score"] in (None, "") else item["pareto_score"], strategy_display_name(item["strategy"], item["hybrid_alpha"])))
        best_pareto = rows_pareto[0] if rows_pareto else None
        if best_loss is not None:
            best_loss["best_by_loss"] = True
        if best_top1 is not None:
            best_top1["best_by_top1"] = True
        if best_pareto is not None:
            best_pareto["best_by_pareto"] = True
        best_table.append(
            {
                "budget_ratio": budget_ratio,
                "best_by_loss": strategy_display_name(best_loss["strategy"], best_loss["hybrid_alpha"]) if best_loss else "",
                "best_by_top1": strategy_display_name(best_top1["strategy"], best_top1["hybrid_alpha"]) if best_top1 else "",
                "best_by_pareto": strategy_display_name(best_pareto["strategy"], best_pareto["hybrid_alpha"]) if best_pareto else "",
            }
        )

    summary_fieldnames = [
        "strategy",
        "hybrid_alpha",
        "budget_ratio",
        "mean_loss_increase",
        "std_loss_increase",
        "mean_continuation_ppl",
        "std_continuation_ppl",
        "mean_top1_agreement",
        "std_top1_agreement",
        "mean_effective_budget_ratio",
        "mean_memory_reduction_ratio",
        "num_prompts",
        "num_seeds",
        "rank_by_loss",
        "rank_by_top1",
        "best_by_loss",
        "best_by_top1",
        "best_by_pareto",
        "pareto_score",
    ]
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    def _find_summary(rows, strategy, budget_ratio, hybrid_alpha=""):
        for row in rows:
            if row["strategy"] == strategy and row["hybrid_alpha"] == hybrid_alpha and row["budget_ratio"] == budget_ratio:
                return row
        return None

    comparison_specs = [
        ("state_norm_keep", "", "random_drop", ""),
        ("state_norm_keep", "", "uniform_drop", ""),
    ]
    for row in summary_rows:
        if row["strategy"] == "hybrid_norm_head":
            comparison_specs.append(("hybrid_norm_head", row["hybrid_alpha"], "state_norm_keep", ""))
        if row["strategy"] == "hybrid_norm_group":
            comparison_specs.append(("hybrid_norm_group", row["hybrid_alpha"], "state_norm_keep", ""))

    comparison_rows = []
    dedup_seen = set()
    for strategy_a, alpha_a, strategy_b, alpha_b in comparison_specs:
        for budget_ratio in sorted({row["budget_ratio"] for row in summary_rows if row["budget_ratio"] is not None}):
            dedup_key = (strategy_a, alpha_a, strategy_b, alpha_b, budget_ratio)
            if dedup_key in dedup_seen:
                continue
            dedup_seen.add(dedup_key)
            rows_a = [row for row in eval_rows if row.get("strategy") == strategy_a and row.get("hybrid_alpha", "") == alpha_a and row["_budget_ratio"] == budget_ratio]
            rows_b = [row for row in eval_rows if row.get("strategy") == strategy_b and row.get("hybrid_alpha", "") == alpha_b and row["_budget_ratio"] == budget_ratio]
            map_a = {(row.get("seed", ""), row.get("prompt_id", "")): row for row in rows_a}
            map_b = {(row.get("seed", ""), row.get("prompt_id", "")): row for row in rows_b}
            common_keys = sorted(set(map_a) & set(map_b))
            if not common_keys:
                continue
            deltas = []
            wins = 0
            a_losses = []
            b_losses = []
            for pair_key in common_keys:
                loss_a = map_a[pair_key]["_loss_increase"]
                loss_b = map_b[pair_key]["_loss_increase"]
                deltas.append(loss_a - loss_b)
                a_losses.append(loss_a)
                b_losses.append(loss_b)
                if loss_a < loss_b:
                    wins += 1
            mean_delta = float(statistics.mean(deltas)) if deltas else ""
            win_rate = float(wins / len(common_keys)) if common_keys else ""
            p_value = ""
            try:
                from scipy.stats import ttest_rel  # type: ignore

                p_value = float(ttest_rel(a_losses, b_losses).pvalue)
            except Exception:
                p_value = ""
            comparison_rows.append(
                {
                    "budget_ratio": budget_ratio,
                    "strategy_a": strategy_display_name(strategy_a, alpha_a),
                    "strategy_b": strategy_display_name(strategy_b, alpha_b),
                    "mean_delta_loss": mean_delta,
                    "win_rate": win_rate,
                    "num_pairs": len(common_keys),
                    "p_value": p_value,
                }
            )

    print("\nTable 1: Quality-memory tradeoff")
    for row in sorted(summary_rows, key=lambda item: (float("inf") if item["budget_ratio"] is None else item["budget_ratio"], item["rank_by_loss"], strategy_display_name(item["strategy"], item["hybrid_alpha"]))):
        print(
            f"budget={row['budget_ratio']} strategy={strategy_display_name(row['strategy'], row['hybrid_alpha'])} "
            f"loss={row['mean_loss_increase']:.6f}±{row['std_loss_increase']:.6f} "
            f"top1={row['mean_top1_agreement']:.6f} eff={row['mean_effective_budget_ratio']:.4f} mem_red={row['mean_memory_reduction_ratio']:.4f} rank={row['rank_by_loss']}"
        )

    print("\nTable 2: Best strategy per budget")
    for row in sorted(best_table, key=lambda item: item["budget_ratio"]):
        print(
            f"budget={row['budget_ratio']} best_by_loss={row['best_by_loss']} "
            f"best_by_top1={row['best_by_top1']} best_by_pareto={row['best_by_pareto']}"
        )

    print("\nTable 3: Win-rate against state_norm_keep")
    for row in comparison_rows:
        print(
            f"budget={row['budget_ratio']} {row['strategy_a']} vs {row['strategy_b']} "
            f"mean_delta_loss={row['mean_delta_loss']} win_rate={row['win_rate']} num_pairs={row['num_pairs']} p_value={row['p_value']}"
        )

    for strategy_key in sorted({(row['strategy'], row['hybrid_alpha']) for row in summary_rows}, key=lambda item: strategy_display_name(item[0], item[1])):
        strategy, hybrid_alpha = strategy_key
        budget_to_loss = {
            row["budget_ratio"]: row["mean_loss_increase"]
            for row in summary_rows
            if row["strategy"] == strategy and row["hybrid_alpha"] == hybrid_alpha and row["mean_loss_increase"] not in (None, "")
        }
        if 0.7 in budget_to_loss and 0.8 in budget_to_loss and 0.9 in budget_to_loss:
            if not (budget_to_loss[0.9] < budget_to_loss[0.8] < budget_to_loss[0.7]):
                print(
                    f"[WARN] no-regression check failed for {strategy_display_name(strategy, hybrid_alpha)}: "
                    f"loss@0.7={budget_to_loss[0.7]} loss@0.8={budget_to_loss[0.8]} loss@0.9={budget_to_loss[0.9]}"
                )

    print(f"Saved summary to {summary_output_path}")
    return summary_rows


def get_state_tensor_bytes(tensor):
    return int(tensor.element_size() * tensor.numel()) if torch.is_tensor(tensor) else 0


def estimate_memory_bytes(packed_cache, mixer_pairs, kept_fraction=None):
    total = 0
    for layer_idx, mixer in mixer_pairs:
        if layer_idx not in packed_cache["key_value_memory_dict"]:
            continue
        conv_state, ssm_state = packed_cache["key_value_memory_dict"][layer_idx]
        total += get_state_tensor_bytes(conv_state) + get_state_tensor_bytes(ssm_state)
    if kept_fraction is None:
        return total
    return int(total * kept_fraction)


def build_synthetic_or_wikitext_samples(args, tokenizer):
    if args.dataset == "synthetic":
        samples = build_synthetic_eval_samples(tokenizer, args.num_prompts, args.prefix_len, args.continuation_len, args.seed)
    else:
        samples = sample_token_chunks(tokenizer, "wikitext2", "validation", args.num_prompts, args.prefix_len, args.continuation_len, args.seed)
    return samples


def validate_sample(sample, args):
    if sample["input_ids"].shape != (1, args.prefix_len + args.continuation_len):
        raise ValueError(f"Bad input_ids shape for prompt {sample['prompt_id']}: {tuple(sample['input_ids'].shape)}")
    if sample["prefix_ids"].shape != (1, args.prefix_len):
        raise ValueError(f"Bad prefix_ids shape for prompt {sample['prompt_id']}: {tuple(sample['prefix_ids'].shape)}")
    if sample["continuation_ids"].shape != (1, args.continuation_len):
        raise ValueError(f"Bad continuation_ids shape for prompt {sample['prompt_id']}: {tuple(sample['continuation_ids'].shape)}")


def ensure_tensor_like_cache_shapes(packed_cache, mixer_pairs):
    for layer_idx, mixer in mixer_pairs:
        if layer_idx not in packed_cache["key_value_memory_dict"]:
            continue
        conv_state, ssm_state = packed_cache["key_value_memory_dict"][layer_idx]
        if not torch.is_tensor(conv_state) or not torch.is_tensor(ssm_state):
            raise TypeError(f"Unsupported cache entry for layer {layer_idx}: {type(conv_state)} / {type(ssm_state)}")


def run_single_invocation(args):
    if args.fast_smoke:
        args.num_prompts = 1
        args.prefix_len = 128
        args.continuation_len = 16
        args.strategies = ["full_state", "random_drop", "uniform_drop", "state_norm_keep", "layer_aware_statecache", "head_aware_statecache"]
        args.budget_ratios = [0.5, 0.8, 1.0]
    args.strategies = normalize_strategies(args.strategies)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_path = build_output_path_for_phase23(args)
    print(f"output_csv path: {output_path}")

    score_info = parse_phase22_tables(args.phase22_layer_csv, args.phase22_head_csv, args.phase22_group_csv)
    score_maps = build_score_maps(score_info)
    if args.dry_run_plan:
        print_dry_run_plan(args, score_info)
        return

    if args.dry_run_samples:
        ensure_local_hf_cache(Path(args.output_dir))
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
        samples = build_synthetic_or_wikitext_samples(args, tokenizer)
        for sample in samples[:3]:
            print(f"sample prompt_id={sample['prompt_id']} input_shape={tuple(sample['input_ids'].shape)} prefix_shape={tuple(sample['prefix_ids'].shape)} continuation_shape={tuple(sample['continuation_ids'].shape)}")
        return

    ensure_local_hf_cache(Path(args.output_dir))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    print(f"Loading model: {MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_loader = build_model(device=device, dtype=torch.float16 if device == "cuda" else torch.float32, local_files_only=False)
    print(f"Model loader: {model_loader}")
    configure_mamba2_for_equivalence(model, disable_mem_eff_path=args.disable_mem_eff_path)

    mixer_pairs = collect_mamba2_mixers(model)
    print(f"Collected {len(mixer_pairs)} Mamba2 mixers")
    if not mixer_pairs:
        raise RuntimeError("No Mamba2 mixers found in model")

    print(f"Loading and tokenizing dataset={args.dataset}")
    eval_samples = build_synthetic_or_wikitext_samples(args, tokenizer)
    if not eval_samples:
        error_row = make_error_row(args, prompt_id=-1, strategy="__setup__", budget_ratio=1.0, error_msg="no_valid_samples")
        write_rows(output_path, [error_row], append=False)
        print(f"Saved results to {output_path}")
        return

    for sample in eval_samples:
        validate_sample(sample, args)

    print(f"Prepared {len(eval_samples)} eval samples")
    if args.dataset == "synthetic":
        print("dataset=synthetic")
    print(f"[info] append_each_prompt={args.append_each_prompt}")
    print(f"[info] debug_budget={args.debug_budget}")
    args.strategies = normalize_strategy_order(args.strategies, args.strategy_order, args.single_strategy_debug)
    strategy_variants = expand_strategy_variants(args.strategies, args.hybrid_alpha, args.anchor_boost)
    anchor_layers = parse_anchor_layers(args.anchor_layers)
    anchor_heads = parse_anchor_heads(args.anchor_heads)
    print(f"[info] strategy_order={args.strategy_order}")
    print(f"[info] ordered_strategies={args.strategies}")
    print(f"[info] strategy_variants={strategy_variants}")

    rows_to_write = []
    wrote_any_row = False
    total_runtime_start = time.perf_counter()
    prompt_executed = False

    try:
        for sample in eval_samples:
            prompt_executed = True
            prompt_id = int(sample["prompt_id"])
            prefix_ids = sample["prefix_ids"]
            continuation_ids = sample["continuation_ids"]
            print(f"prompt_id={prompt_id} prefix_len={args.prefix_len} continuation_len={args.continuation_len}")

            started_row = {
                "dataset": sample["dataset"],
                "strategy": "__prompt__",
                "budget_ratio": 1.0,
                "effective_budget_ratio": 1.0,
                "prompt_id": prompt_id,
                "prefix_len": args.prefix_len,
                "continuation_len": args.continuation_len,
                "full_loss": "",
                "continuation_loss": "",
                "loss_increase": "",
                "full_ppl": "",
                "continuation_ppl": "",
                "logits_mse_to_full": "",
                "top1_agreement_to_full": "",
                "original_state_elements": "",
                "kept_state_elements": "",
                "estimated_state_memory_bytes": "",
                "memory_reduction_ratio": "",
                "average_decode_latency_ms": "",
                "tokens_per_second": "",
                "status": "started",
                "error_msg": "",
            }
            rows_to_write.append(started_row)
            wrote_any_row = True
            if args.append_each_prompt:
                write_rows(output_path, rows_to_write, append=True)
                rows_to_write = []

            try:
                reference_prefill_logits, reference_cache = prefill_prefix(
                    model, prefix_ids, args.prefix_len + args.continuation_len, device, args.cache_device
                )
                ensure_tensor_like_cache_shapes(reference_cache, mixer_pairs)
                clean_cache = clone_cache_deep(reference_cache)
                clean_fp_before = cache_fingerprint_signature(clean_cache)
                full_logits, full_latency_ms, token_count = run_full_state_from_clean(
                    model, clean_cache, reference_prefill_logits, continuation_ids, device
                )
                full_loss, full_ppl = compute_loss_and_ppl(full_logits, continuation_ids)
                original_state_elements = 0
                for layer_idx, mixer in mixer_pairs:
                    if layer_idx not in clean_cache["key_value_memory_dict"]:
                        continue
                    conv_state, ssm_state = clean_cache["key_value_memory_dict"][layer_idx]
                    original_state_elements += int(conv_state.numel() + ssm_state.numel())
                state_memory_bytes = estimate_memory_bytes(clean_cache, mixer_pairs, kept_fraction=1.0)
                total_cache_elements = original_state_elements
                manageable_state_elements = original_state_elements
                kept_manageable_elements = original_state_elements
                unmanaged_elements = 0
                kept_state_elements = original_state_elements
                full_repeat_check = True

                if args.validate_full_repeat:
                    repeat_logits_before, _, _ = run_full_state_from_clean(
                        model, clean_cache, reference_prefill_logits, continuation_ids, device
                    )
                    full_repeat_check = torch.equal(full_logits, repeat_logits_before)
                    if not full_repeat_check:
                        raise RuntimeError("status=full_repeat_mismatch: full_state pre-check failed")

                full_row = make_result_row(
                    args,
                    sample,
                    "full_state",
                    1.0,
                    full_loss,
                    full_loss,
                    full_ppl,
                    full_ppl,
                    0.0,
                    1.0,
                    original_state_elements,
                    total_cache_elements,
                    original_state_elements,
                    original_state_elements,
                    0,
                    kept_state_elements,
                    state_memory_bytes,
                    full_latency_ms,
                    (args.continuation_len / (full_latency_ms / 1000.0)) if full_latency_ms > 0 else 0.0,
                    False,
                    True,
                    True,
                    True,
                    full_repeat_check,
                    "",
                    args.strategy_order,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    1.0,
                    0.0,
                    "",
                    "",
                    "functional_pass",
                    "",
                    1.0,
                    0.0,
                )
                rows_to_write.append(full_row)

                strategy_results = {}
                budget_list = [float(x) for x in args.budget_ratios]
                norm_keep_hash_by_budget = {}
                for strategy, hybrid_alpha, anchor_boost in strategy_variants:
                    strategy_key = (strategy, hybrid_alpha, anchor_boost)
                    keep_sets_by_budget = {}
                    strategy_results[strategy_key] = []
                    for budget_ratio in budget_list:
                        if strategy == "full_state" and budget_ratio != 1.0:
                            continue
                        if strategy == "full_state":
                            continue
                        try:
                            clean_fp_before_strategy = cache_fingerprint_signature(clean_cache)
                            work_cache = clone_cache_deep(clean_cache)
                            alias_check_pass, alias_pairs = check_cache_alias(clean_cache, work_cache, debug=args.debug_cache_alias)
                            if not alias_check_pass:
                                raise RuntimeError(f"status=cache_contaminated: shared storage detected {alias_pairs[:3]}")

                            modified_cache, budget_stats = exact_budget_allocate(
                                strategy,
                                budget_ratio,
                                work_cache,
                                mixer_pairs,
                                score_maps,
                                debug_budget=args.debug_budget,
                                seed=args.seed + prompt_id,
                                hybrid_alpha=hybrid_alpha,
                                anchor_layers=anchor_layers,
                                anchor_heads=anchor_heads,
                                anchor_min_keep_ratio=args.anchor_min_keep_ratio,
                                anchor_mode=args.anchor_mode,
                                anchor_boost=(anchor_boost or 0.0),
                            )
                            alias_check_pass_2, alias_pairs_2 = check_cache_alias(clean_cache, modified_cache, debug=args.debug_cache_alias)
                            if not alias_check_pass_2:
                                raise RuntimeError(f"status=cache_contaminated: post-alloc shared storage detected {alias_pairs_2[:3]}")
                            total_cache_elements = budget_stats["total_cache_elements"]
                            manageable_state_elements = budget_stats["manageable_state_elements"]
                            kept_manageable_elements = budget_stats["kept_manageable_elements"]
                            unmanaged_elements = budget_stats["unmanaged_elements"]
                            effective_budget_ratio = budget_stats["effective_budget_ratio"]
                            budget_abs_error = budget_stats["budget_abs_error"]
                            estimated_state_memory_bytes = budget_stats["estimated_state_memory_bytes"]
                            kept_state_elements = budget_stats["kept_state_elements"]
                            candidate_logits, latency_ms, tokens = decode_continuation(model, modified_cache, reference_prefill_logits, continuation_ids, device)
                            continuation_loss, continuation_ppl = compute_loss_and_ppl(candidate_logits, continuation_ids)
                            logits_mse, top1_agreement, _ = compute_logits_metrics(full_logits, candidate_logits)
                            status = budget_stats["status"]
                            keep_ids = {_unit_identity(unit) for unit in budget_stats.get("selected_units", [])}
                            keep_sets_by_budget[float(budget_ratio)] = keep_ids
                            nested_budget_check = True
                            nested_error = ""
                            if args.validate_nested_budgets:
                                nested_budget_check, nested_error = validate_nested_keep_sets(strategy, keep_sets_by_budget, keep_sets_by_budget.keys())
                                if not nested_budget_check:
                                    status = "non_nested_mask"
                            priority_direction_check, priority_error = validate_priority_direction(
                                strategy,
                                budget_stats.get("selected_units", []),
                                budget_stats.get("dropped_units", []),
                            )
                            if strategy in {"state_norm_keep", "hybrid_norm_layer", "hybrid_norm_head", "hybrid_norm_layer_head", "anchor_protected_state_norm"} and not priority_direction_check:
                                status = "priority_direction_error"
                            clean_fp_after_strategy = cache_fingerprint_signature(clean_cache)
                            clean_cache_changed = fingerprint_changed(clean_fp_before_strategy, clean_fp_after_strategy)
                            if args.debug_cache_alias:
                                print(f"[cache_alias] clean fingerprint before={summarize_fingerprint(clean_fp_before_strategy)}")
                                print(f"[cache_alias] clean fingerprint after={summarize_fingerprint(clean_fp_after_strategy)}")
                            if clean_cache_changed:
                                raise RuntimeError("status=cache_contaminated: clean cache fingerprint changed")

                            full_repeat_check_value = True
                            if args.validate_full_repeat:
                                repeat_logits_after, _, _ = run_full_state_from_clean(
                                    model, clean_cache, reference_prefill_logits, continuation_ids, device
                                )
                                full_repeat_check_value = torch.equal(full_logits, repeat_logits_after)
                                if not full_repeat_check_value:
                                    status = "full_repeat_mismatch"

                            error_parts = []
                            if nested_error:
                                error_parts.append(nested_error)
                            if priority_error and strategy in {"state_norm_keep", "hybrid_norm_layer", "hybrid_norm_head", "hybrid_norm_group", "hybrid_norm_layer_head", "hybrid_norm_layer_head_group", "anchor_protected_state_norm"}:
                                error_parts.append(priority_error)
                            final_priority_changed_vs_norm = ""
                            if strategy == "state_norm_keep":
                                norm_keep_hash_by_budget[float(budget_ratio)] = budget_stats.get("keep_set_hash", "")
                            elif strategy.startswith("hybrid_") and hybrid_alpha is not None:
                                baseline_hash = norm_keep_hash_by_budget.get(float(budget_ratio), "")
                                current_hash = budget_stats.get("keep_set_hash", "")
                                final_priority_changed_vs_norm = (current_hash != baseline_hash) if baseline_hash else ""
                                if float(hybrid_alpha) > 0 and bool(budget_stats.get("sensitivity_valid", False)) and baseline_hash and current_hash == baseline_hash:
                                    error_parts.append("hybrid_no_effect")
                            row = make_result_row(
                                args,
                                sample,
                                strategy,
                                budget_ratio,
                                full_loss,
                                continuation_loss,
                                full_ppl,
                                continuation_ppl,
                                logits_mse,
                                top1_agreement,
                                original_state_elements,
                                total_cache_elements,
                                manageable_state_elements,
                                kept_manageable_elements,
                                unmanaged_elements,
                                kept_state_elements,
                                estimated_state_memory_bytes,
                                latency_ms,
                                (args.continuation_len / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0,
                                clean_cache_changed,
                                alias_check_pass and alias_check_pass_2,
                                nested_budget_check,
                                priority_direction_check,
                                full_repeat_check_value,
                                budget_stats.get("keep_set_hash", ""),
                                args.strategy_order,
                                hybrid_alpha if hybrid_alpha is not None else "",
                                args.anchor_min_keep_ratio if strategy == "anchor_protected_state_norm" else "",
                                budget_stats.get("norm_score", ""),
                                budget_stats.get("sensitivity_score", ""),
                                budget_stats.get("final_priority_score", ""),
                                budget_stats.get("sensitivity_valid", ""),
                                budget_stats.get("sensitivity_nonzero_count", ""),
                                budget_stats.get("sensitivity_max", ""),
                                budget_stats.get("sensitivity_mean_positive", ""),
                                budget_stats.get("sensitivity_source_file", ""),
                                budget_stats.get("sensitivity_score_std", ""),
                                final_priority_changed_vs_norm,
                                budget_stats.get("kept_mean_priority", ""),
                                budget_stats.get("dropped_mean_priority", ""),
                                budget_stats.get("priority_top10", ""),
                                budget_stats.get("priority_bottom10", ""),
                                status,
                                "; ".join(error_parts),
                                effective_budget_ratio,
                                budget_abs_error,
                            )
                            strategy_results[strategy_key].append(row)
                            rows_to_write.append(row)
                            del work_cache
                            del modified_cache
                        except Exception as exc:
                            import traceback

                            rows_to_write.append(
                                make_error_row(
                                    args,
                                    prompt_id,
                                    strategy,
                                    budget_ratio,
                                    error_msg=f"{exc}\n{traceback.format_exc()}",
                                    status=(str(exc).split("status=")[-1].split(":")[0] if "status=" in str(exc) else "error"),
                                    hybrid_alpha=hybrid_alpha if hybrid_alpha is not None else "",
                                    anchor_min_keep_ratio=args.anchor_min_keep_ratio if strategy == "anchor_protected_state_norm" else "",
                                    sensitivity_source_file=(score_maps["group_info"].get("path", "") if strategy in {"hybrid_norm_group", "hybrid_norm_layer_head_group"} else score_maps["head_info"].get("path", "") if strategy == "hybrid_norm_head" else score_maps["layer_info"].get("path", "") if strategy in {"hybrid_norm_layer", "hybrid_norm_layer_head"} else ""),
                                )
                            )
                            if args.append_each_prompt:
                                write_rows(output_path, rows_to_write, append=True)
                                rows_to_write = []
                            if args.append_each_prompt is False and args.append_each_prompt is not None:
                                pass

                if args.validate_full_repeat:
                    repeat_logits_after_all, _, _ = run_full_state_from_clean(
                        model, clean_cache, reference_prefill_logits, continuation_ids, device
                    )
                    if not torch.equal(full_logits, repeat_logits_after_all):
                        raise RuntimeError("status=full_repeat_mismatch: full_state post-check failed")
            except Exception as exc:
                import traceback

                err_row = make_error_row(
                    args,
                    prompt_id,
                    "__prompt__",
                    1.0,
                    error_msg=f"{exc}\n{traceback.format_exc()}",
                )
                rows_to_write.append(err_row)

            if rows_to_write:
                write_rows(output_path, rows_to_write, append=True)
                rows_to_write = []

        if not prompt_executed:
            write_rows(output_path, [make_error_row(args, -1, "__main__", 1.0, "prompt loop did not execute")], append=False)
        elif not wrote_any_row:
            write_rows(output_path, [make_error_row(args, -1, "__main__", 1.0, "prompt loop did not execute")], append=False)
    except Exception as exc:
        import traceback

        write_rows(
            output_path,
            [make_error_row(args, -1, "__main__", 1.0, f"{exc}\n{traceback.format_exc()}")],
            append=False,
        )
        raise

    total_runtime_ms = (time.perf_counter() - total_runtime_start) * 1000.0
    print(f"Saved results to {output_path}")
    print(f"[summary] total_runtime_ms={total_runtime_ms:.2f}")
    return output_path


def main():
    args = parse_args()
    if args.fast_smoke:
        args.num_prompts = 1
        args.prefix_len = 128
        args.continuation_len = 16
        args.strategies = ["full_state", "random_drop", "uniform_drop", "state_norm_keep", "layer_aware_statecache", "head_aware_statecache"]
        args.budget_ratios = [0.5, 0.8, 1.0]
    args = resolve_preset(args)
    args.strategies = normalize_strategies(args.strategies)
    seed_list = resolve_seed_list(args)

    if args.dry_run_plan:
        score_info = parse_phase22_tables(args.phase22_layer_csv, args.phase22_head_csv, args.phase22_group_csv)
        print_dry_run_plan(args, score_info)
        return

    output_path = None
    for index, seed in enumerate(seed_list):
        seed_args = copy.deepcopy(args)
        seed_args.seed = int(seed)
        if output_path is not None:
            seed_args.output_csv = str(output_path)
        print(f"\n=== Phase23 seed {seed_args.seed} ({index + 1}/{len(seed_list)}) ===")
        output_path = run_single_invocation(seed_args)

    if output_path is not None:
        build_phase23_summary(output_path)


if __name__ == "__main__":
    main()
