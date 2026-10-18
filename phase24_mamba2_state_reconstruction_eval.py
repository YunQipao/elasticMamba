from __future__ import annotations

"""
Phase24 evaluates compressed recurrent state storage and restoration quality.
It does not claim per-token decode acceleration unless a quantized/reduced
compute kernel is implemented.
"""

import argparse
import bisect
import copy
import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

torch = None
F = None
AutoTokenizer = None
MODEL_NAME = "state-spaces/mamba2-130m"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"


DEFAULT_STRATEGIES = [
    "full_state",
    "random_zero_drop",
    "uniform_zero_drop",
    "state_norm_zero_drop",
    "hybrid_zero_drop",
    "state_quant_int8",
    "state_quant_int4",
    "lowrank_reconstruct",
    "priority_mixed_quant",
    "lowrank_reconstruct_plus_int8",
    "residual_lowrank_plus_int8",
]
DEFAULT_BUDGET_RATIOS = [0.6, 0.7, 0.8, 0.9]
DEFAULT_RANKS = [4, 8, 16]
DEFAULT_OUTPUT_STEM = "phase24_mamba2_state_reconstruction_eval"
DEFAULT_SUMMARY_STEM = "phase24_checkpoint_storage_summary"
DEFAULT_PAIRED_STEM = "phase24_state_reconstruction_paired_comparison"
DEFAULT_PAPER_TABLE_STEM = "phase24_state_reconstruction_paper_table"
DEFAULT_PARETO_STEM = "phase24_state_reconstruction_pareto"
PHASE24_SUMMARY_TEXT = (
    "Phase24 evaluates compressed recurrent state storage and restoration quality. "
    "It does not claim per-token decode acceleration unless a quantized/reduced compute kernel is implemented."
)
ONE_GB_BYTES = 1024 ** 3
MIXED_SELECTOR_METHOD = "fast_prefix_binary_search"

CSV_FIELDNAMES = [
    "run_id",
    "seed",
    "dataset",
    "strategy",
    "reconstruction_method",
    "rank",
    "hybrid_alpha",
    "budget_ratio",
    "budget_mode",
    "effective_budget_ratio",
    "summary",
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
    "reconstruction_mse",
    "quantization_bits",
    "full_precision_fraction",
    "quantized_fraction",
    "quantization_scale_mean",
    "quantization_scale_max",
    "quantization_mse",
    "fallback_reason",
    "original_state_elements",
    "kept_state_elements",
    "reconstructed_state_elements",
    "original_state_bytes",
    "compressed_state_bytes",
    "estimated_state_memory_bytes",
    "fp16_state_bytes",
    "int8_state_bytes",
    "int4_state_bytes",
    "scale_bytes",
    "lowrank_factor_bytes",
    "residual_bytes",
    "compression_ratio",
    "memory_reduction_ratio",
    "max_sessions_full_state_under_1GB",
    "max_sessions_compressed_under_1GB",
    "session_capacity_gain",
    "checkpoint_compress_ms",
    "checkpoint_restore_ms",
    "transform_total_ms",
    "continuation_decode_ms",
    "reconstruction_overhead_ms",
    "priority_build_ms",
    "mask_build_ms",
    "lowrank_decompose_ms",
    "reconstruct_apply_ms",
    "quantize_ms",
    "decode_ms",
    "average_decode_latency_ms",
    "tokens_per_second",
    "mixed_selector_method",
    "mixed_selector_num_units",
    "mixed_selector_elapsed_ms",
    "mixed_selector_budget_abs_error",
    "mixed_selector_fp16_units",
    "mixed_selector_int8_units",
    "mixed_selector_int4_units",
    "mixed_selector_fp16_bytes",
    "mixed_selector_int8_bytes",
    "mixed_selector_int4_bytes",
    "budget_under_min",
    "budget_over_max",
    "clean_cache_changed",
    "alias_check_pass",
    "full_repeat_check",
    "status",
    "error_msg",
]

SUMMARY_FIELDNAMES = [
    "run_id",
    "strategy",
    "rank",
    "hybrid_alpha",
    "budget_ratio",
    "quantization_bits",
    "summary",
    "mean_effective_budget_ratio",
    "mean_loss_increase",
    "std_loss_increase",
    "mean_continuation_ppl",
    "mean_top1_agreement",
    "std_top1_agreement",
    "mean_reconstruction_mse",
    "mean_quantization_mse",
    "mean_compressed_state_bytes",
    "mean_memory_reduction_ratio",
    "mean_checkpoint_compress_ms",
    "mean_checkpoint_restore_ms",
    "mean_transform_total_ms",
    "mean_continuation_decode_ms",
    "mean_session_capacity_gain",
    "num_prompts",
    "num_seeds",
    "win_rate_vs_state_quant_int8",
    "win_rate_vs_hybrid_zero_drop",
    "rank_by_loss",
    "rank_by_pareto",
    "warning",
]

PAIRED_FIELDNAMES = [
    "run_id",
    "method_a",
    "method_b",
    "rank",
    "budget_ratio",
    "num_pairs",
    "mean_delta_loss",
    "win_rate_loss",
    "mean_delta_top1",
    "win_rate_top1",
    "mean_delta_reconstruction_mse",
    "mean_delta_memory_reduction_ratio",
]

PAPER_TABLE_FIELDNAMES = [
    "budget_ratio",
    "method",
    "rank",
    "quantization_bits",
    "summary",
    "mean_loss_increase",
    "std_loss_increase",
    "mean_top1_agreement",
    "std_top1_agreement",
    "mean_reconstruction_mse",
    "mean_quantization_mse",
    "mean_compressed_state_bytes",
    "mean_memory_reduction_ratio",
    "mean_checkpoint_compress_ms",
    "mean_checkpoint_restore_ms",
    "mean_continuation_decode_ms",
    "mean_session_capacity_gain",
    "win_rate_vs_hybrid_zero_drop",
    "win_rate_vs_state_quant_int8",
    "rank_by_loss",
    "rank_by_pareto",
]

PARETO_FIELDNAMES = [
    "run_id",
    "budget_ratio",
    "method",
    "rank",
    "quantization_bits",
    "summary",
    "mean_effective_budget_ratio",
    "mean_loss_increase",
    "mean_top1_agreement",
    "mean_compressed_state_bytes",
    "mean_memory_reduction_ratio",
    "mean_checkpoint_compress_ms",
    "mean_checkpoint_restore_ms",
    "is_pareto_frontier",
    "pareto_layer",
]

RECONSTRUCTION_METHOD_MAP = {
    "full_state": "identity",
    "random_zero_drop": "zero_drop",
    "uniform_zero_drop": "zero_drop",
    "state_norm_zero_drop": "zero_drop",
    "hybrid_zero_drop": "zero_drop",
    "state_quant_int8": "state_quantization",
    "state_quant_int4": "state_quantization",
    "hybrid_zero_drop_plus_int8": "zero_drop_plus_quant",
    "mean_reconstruct": "mean",
    "mean_scale_reconstruct": "mean_scale",
    "lowrank_reconstruct": "lowrank",
    "priority_mixed_quant": "priority_mixed_quant",
    "lowrank_reconstruct_plus_int8": "lowrank_plus_quant",
    "residual_lowrank_plus_int8": "residual_lowrank_plus_int8",
    "linear_head_reconstruct": "linear_head",
    "zero_drop": "zero_drop",
    "state_norm_keep": "zero_drop",
    "hybrid_norm_head": "zero_drop",
}

STRATEGY_ALIASES = {
    "zero_drop": "hybrid_zero_drop",
    "hybrid_norm_head": "hybrid_zero_drop",
    "state_norm_keep": "state_norm_zero_drop",
}

SUPPORTED_STRATEGIES = {
    "full_state",
    "random_zero_drop",
    "uniform_zero_drop",
    "state_norm_zero_drop",
    "hybrid_zero_drop",
    "state_quant_int8",
    "state_quant_int4",
    "hybrid_zero_drop_plus_int8",
    "mean_reconstruct",
    "mean_scale_reconstruct",
    "lowrank_reconstruct",
    "priority_mixed_quant",
    "lowrank_reconstruct_plus_int8",
    "residual_lowrank_plus_int8",
    "linear_head_reconstruct",
    *STRATEGY_ALIASES.keys(),
}

HYBRID_PRIORITY_STRATEGIES = {
    "hybrid_zero_drop",
    "state_quant_int8",
    "state_quant_int4",
    "hybrid_zero_drop_plus_int8",
    "mean_reconstruct",
    "mean_scale_reconstruct",
    "lowrank_reconstruct",
    "lowrank_reconstruct_plus_int8",
    "priority_mixed_quant",
    "linear_head_reconstruct",
}
RANDOM_PRIORITY_STRATEGIES = {"random_zero_drop"}
UNIFORM_PRIORITY_STRATEGIES = {"uniform_zero_drop"}
STATE_NORM_PRIORITY_STRATEGIES = {"state_norm_zero_drop"}
RECONSTRUCTION_STRATEGIES = {
    "mean_reconstruct",
    "mean_scale_reconstruct",
    "lowrank_reconstruct",
    "lowrank_reconstruct_plus_int8",
    "residual_lowrank_plus_int8",
    "linear_head_reconstruct",
}
LOWRANK_STRATEGIES = {"lowrank_reconstruct", "lowrank_reconstruct_plus_int8"}
RANK_ONLY_STRATEGIES = {"residual_lowrank_plus_int8"}
QUANTIZATION_STRATEGIES = {
    "state_quant_int8",
    "state_quant_int4",
    "hybrid_zero_drop_plus_int8",
    "lowrank_reconstruct_plus_int8",
    "priority_mixed_quant",
    "residual_lowrank_plus_int8",
}
MIXED_QUANTIZATION_STRATEGIES = {"state_quant_int8", "state_quant_int4"}
KEPT_QUANTIZATION_STRATEGIES = {"hybrid_zero_drop_plus_int8"}
LOWRANK_FACTOR_INT8_STRATEGIES = {"lowrank_reconstruct_plus_int8"}
PRIORITY_MIXED_STRATEGIES = {"priority_mixed_quant"}
RESIDUAL_CORRECTION_STRATEGIES = {"residual_lowrank_plus_int8"}


def require_runtime_deps():
    global torch, F, AutoTokenizer
    global MODEL_NAME, TOKENIZER_NAME
    global build_model, collect_mamba2_mixers, configure_mamba2_for_equivalence, ensure_local_hf_cache
    global build_priority_lists_for_units, build_score_maps, build_synthetic_or_wikitext_samples
    global cache_fingerprint_signature, check_cache_alias, clone_cache_deep, collect_cache_inventory
    global compute_logits_metrics, compute_loss_and_ppl, compute_state_norm_scores, decode_continuation
    global ensure_tensor_like_cache_shapes, fingerprint_changed, mean_std, parse_phase22_tables
    global prefill_prefix, run_full_state_from_clean, safe_float, safe_int, select_units_exact_budget, validate_sample

    if torch is not None:
        return

    import torch as _torch
    import torch.nn.functional as _F
    from transformers import AutoTokenizer as _AutoTokenizer

    from phase12_mamba2_head_permutation_equivalence import (
        MODEL_NAME as _MODEL_NAME,
        TOKENIZER_NAME as _TOKENIZER_NAME,
        build_model as _build_model,
        collect_mamba2_mixers as _collect_mamba2_mixers,
        configure_mamba2_for_equivalence as _configure_mamba2_for_equivalence,
        ensure_local_hf_cache as _ensure_local_hf_cache,
    )
    from phase23_mamba2_state_cache_compression_eval import (
        build_priority_lists_for_units as _build_priority_lists_for_units,
        build_score_maps as _build_score_maps,
        build_synthetic_or_wikitext_samples as _build_synthetic_or_wikitext_samples,
        cache_fingerprint_signature as _cache_fingerprint_signature,
        check_cache_alias as _check_cache_alias,
        clone_cache_deep as _clone_cache_deep,
        collect_cache_inventory as _collect_cache_inventory,
        compute_logits_metrics as _compute_logits_metrics,
        compute_loss_and_ppl as _compute_loss_and_ppl,
        compute_state_norm_scores as _compute_state_norm_scores,
        decode_continuation as _decode_continuation,
        ensure_tensor_like_cache_shapes as _ensure_tensor_like_cache_shapes,
        fingerprint_changed as _fingerprint_changed,
        mean_std as _mean_std,
        parse_phase22_tables as _parse_phase22_tables,
        prefill_prefix as _prefill_prefix,
        run_full_state_from_clean as _run_full_state_from_clean,
        safe_float as _safe_float,
        safe_int as _safe_int,
        select_units_exact_budget as _select_units_exact_budget,
        validate_sample as _validate_sample,
    )

    torch = _torch
    F = _F
    AutoTokenizer = _AutoTokenizer
    MODEL_NAME = _MODEL_NAME
    TOKENIZER_NAME = _TOKENIZER_NAME
    build_model = _build_model
    collect_mamba2_mixers = _collect_mamba2_mixers
    configure_mamba2_for_equivalence = _configure_mamba2_for_equivalence
    ensure_local_hf_cache = _ensure_local_hf_cache
    build_priority_lists_for_units = _build_priority_lists_for_units
    build_score_maps = _build_score_maps
    build_synthetic_or_wikitext_samples = _build_synthetic_or_wikitext_samples
    cache_fingerprint_signature = _cache_fingerprint_signature
    check_cache_alias = _check_cache_alias
    clone_cache_deep = _clone_cache_deep
    collect_cache_inventory = _collect_cache_inventory
    compute_logits_metrics = _compute_logits_metrics
    compute_loss_and_ppl = _compute_loss_and_ppl
    compute_state_norm_scores = _compute_state_norm_scores
    decode_continuation = _decode_continuation
    ensure_tensor_like_cache_shapes = _ensure_tensor_like_cache_shapes
    fingerprint_changed = _fingerprint_changed
    mean_std = _mean_std
    parse_phase22_tables = _parse_phase22_tables
    prefill_prefix = _prefill_prefix
    run_full_state_from_clean = _run_full_state_from_clean
    safe_float = _safe_float
    safe_int = _safe_int
    select_units_exact_budget = _select_units_exact_budget
    validate_sample = _validate_sample

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def load_csv_rows(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def summarize_phase22_scores(rows, value_field: str, key_fields: list[str]):
    valid = []
    for row in rows:
        if value_field not in row:
            continue
        try:
            score = float(row[value_field])
            _ = tuple(int(row[field]) for field in key_fields if field in row)
        except Exception:
            continue
        if math.isnan(score) or score <= 0:
            continue
        valid.append(row)
    return valid


def dry_run_score_info(args):
    layer_rows = load_csv_rows(args.phase22_layer_csv)
    head_rows = load_csv_rows(args.phase22_head_csv)
    group_rows = load_csv_rows(args.phase22_group_csv)
    return {
        "layer_scores": summarize_phase22_scores(layer_rows, "loss_increase", ["layer_idx"]),
        "head_scores": summarize_phase22_scores(head_rows, "loss_increase", ["layer_idx", "head_id"]),
        "group_scores": summarize_phase22_scores(group_rows, "loss_increase", ["layer_idx", "head_id", "channel_group_id"]),
    }


def local_safe_float(value):
    try:
        if value is None or value == "":
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    except Exception:
        return None


def local_safe_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def local_mean_std(values):
    if not values:
        return "", ""
    if len(values) == 1:
        return float(values[0]), 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return float(mean), float(math.sqrt(max(variance, 0.0)))


def format_exception_message(exc: Exception, max_trace_lines: int = 20):
    import traceback

    exc_type = type(exc).__name__
    tb_lines = traceback.format_exc().splitlines()
    tb_preview = tb_lines[:max_trace_lines]
    return "\n".join([f"{exc_type}: {exc}", *tb_preview])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "synthetic"])
    parser.add_argument("--num_prompts", type=int, default=16)
    parser.add_argument("--prefix_len", type=int, default=512)
    parser.add_argument("--continuation_len", type=int, default=64)
    parser.add_argument("--budget_ratios", nargs="+", type=float, default=DEFAULT_BUDGET_RATIOS)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--hybrid_alpha", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--rank", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase22_layer_csv", type=str, default="outputs/phase22_latest_layer_sensitivity.csv")
    parser.add_argument("--phase22_head_csv", type=str, default="outputs/phase22_latest_head_sensitivity.csv")
    parser.add_argument("--phase22_group_csv", type=str, default="outputs/phase22_latest_group_sensitivity.csv")
    parser.add_argument("--cache_device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--budget_mode", type=str, default="effective_storage", choices=["retained_only", "effective_storage"])
    parser.add_argument("--disable_mem_eff_path", action="store_true")
    parser.add_argument("--append_each_prompt", action="store_true")
    parser.add_argument("--dry_run_plan", action="store_true")
    parser.add_argument("--validate_full_repeat", action="store_true")
    parser.add_argument("--validate_mixed_quant_selector", action="store_true")
    parser.add_argument("--mixed_quant_validate_k", type=int, default=128)
    parser.add_argument("--debug_reconstruction", action="store_true")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def normalize_strategies(strategies):
    normalized = []
    for strategy in strategies:
        strategy = STRATEGY_ALIASES.get(strategy, strategy)
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")
        if strategy not in normalized:
            normalized.append(strategy)
    return normalized


def resolve_seed_list(args):
    seeds = [int(seed) for seed in (args.seeds or [args.seed])]
    deduped = []
    for seed in seeds:
        if seed not in deduped:
            deduped.append(seed)
    return deduped


def expand_strategy_variants(strategies, hybrid_alphas, ranks):
    expanded = []
    for strategy in strategies:
        if strategy == "full_state":
            expanded.append((strategy, None, None))
        elif strategy in LOWRANK_STRATEGIES:
            for alpha in hybrid_alphas:
                for rank in ranks:
                    expanded.append((strategy, float(alpha), int(rank)))
        elif strategy in RANK_ONLY_STRATEGIES:
            for rank in ranks:
                expanded.append((strategy, None, int(rank)))
        elif strategy in HYBRID_PRIORITY_STRATEGIES:
            for alpha in hybrid_alphas:
                expanded.append((strategy, float(alpha), None))
        else:
            expanded.append((strategy, None, None))
    return expanded


def build_run_id(args):
    config = {
        "dataset": args.dataset,
        "num_prompts": args.num_prompts,
        "prefix_len": args.prefix_len,
        "continuation_len": args.continuation_len,
        "budget_ratios": list(args.budget_ratios),
        "strategies": list(args.strategies),
        "hybrid_alpha": list(args.hybrid_alpha),
        "rank": list(args.rank),
        "seeds": list(args.seeds or []),
        "budget_mode": args.budget_mode,
        "cache_device": args.cache_device,
        "disable_mem_eff_path": bool(args.disable_mem_eff_path),
    }
    return hashlib.md5(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def build_output_paths(args):
    run_id = build_run_id(args)
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.output_dir) / f"{DEFAULT_OUTPUT_STEM}__{run_id}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_dir) / f"{DEFAULT_SUMMARY_STEM}__{run_id}.csv"
    paired_path = Path(args.output_dir) / f"{DEFAULT_PAIRED_STEM}__{run_id}.csv"
    paper_table_path = Path(args.output_dir) / f"{DEFAULT_PAPER_TABLE_STEM}__{run_id}.csv"
    pareto_path = Path(args.output_dir) / f"{DEFAULT_PARETO_STEM}__{run_id}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    return run_id, output_path, summary_path, paired_path, paper_table_path, pareto_path


def clear_previous_outputs(*paths: Path):
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def print_dry_run_plan(args, score_info):
    expanded = expand_strategy_variants(args.strategies, args.hybrid_alpha, args.rank)
    seed_list = resolve_seed_list(args)
    non_full = [variant for variant in expanded if variant[0] != "full_state"]
    estimated_runs = (len(seed_list) * args.num_prompts) + (len(non_full) * len(args.budget_ratios) * len(seed_list) * args.num_prompts)
    print("phase24 dry run plan")
    print(f"  summary                 : {PHASE24_SUMMARY_TEXT}")
    print(f"  dataset                 : {args.dataset}")
    print(f"  seeds                   : {seed_list}")
    print(f"  num_prompts             : {args.num_prompts}")
    print(f"  prefix_len              : {args.prefix_len}")
    print(f"  continuation_len        : {args.continuation_len}")
    print(f"  budget_ratios           : {args.budget_ratios}")
    print(f"  budget_mode             : {args.budget_mode}")
    print(f"  strategies              : {args.strategies}")
    print(f"  expanded strategies     : {expanded}")
    print(f"  hybrid_alpha            : {args.hybrid_alpha}")
    print(f"  rank                    : {args.rank}")
    print(f"  cache_device            : {args.cache_device}")
    print(f"  validate_full_repeat    : {args.validate_full_repeat}")
    print(f"  validate_mixed_quant    : {args.validate_mixed_quant_selector}")
    print(f"  mixed_quant_validate_k  : {args.mixed_quant_validate_k}")
    print(f"  append_each_prompt      : {args.append_each_prompt}")
    print(f"  disable_mem_eff_path    : {args.disable_mem_eff_path}")
    print(f"  estimated runs          : {estimated_runs}")
    print(f"  phase22 layer csv exists: {Path(args.phase22_layer_csv).exists()}")
    print(f"  phase22 head csv exists : {Path(args.phase22_head_csv).exists()}")
    print(f"  phase22 group csv exists: {Path(args.phase22_group_csv).exists()}")
    print(f"  valid layer scores      : {len(score_info['layer_scores'])}")
    print(f"  valid head scores       : {len(score_info['head_scores'])}")
    print(f"  valid group scores      : {len(score_info['group_scores'])}")
    print("  model load              : skipped")


def write_rows(output_path: Path, rows: list[dict[str, Any]], append: bool):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and output_path.exists() else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def write_summary(summary_path: Path, rows: list[dict[str, Any]]):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_named_rows(output_path: Path, rows: list[dict[str, Any]], fieldnames: list[str]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_error_row(args, run_id, prompt_id, strategy, budget_ratio, error_msg, hybrid_alpha="", rank="", status="error"):
    strategy = STRATEGY_ALIASES.get(strategy, strategy)
    return {
        "run_id": run_id,
        "seed": getattr(args, "seed", ""),
        "dataset": args.dataset,
        "strategy": strategy,
        "reconstruction_method": RECONSTRUCTION_METHOD_MAP.get(strategy, ""),
        "rank": rank,
        "hybrid_alpha": hybrid_alpha,
        "budget_ratio": budget_ratio,
        "budget_mode": getattr(args, "budget_mode", ""),
        "effective_budget_ratio": "",
        "summary": PHASE24_SUMMARY_TEXT,
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
        "reconstruction_mse": "",
        "quantization_bits": "",
        "full_precision_fraction": "",
        "quantized_fraction": "",
        "quantization_scale_mean": "",
        "quantization_scale_max": "",
        "quantization_mse": "",
        "fallback_reason": "",
        "original_state_elements": "",
        "kept_state_elements": "",
        "reconstructed_state_elements": "",
        "original_state_bytes": "",
        "compressed_state_bytes": "",
        "estimated_state_memory_bytes": "",
        "fp16_state_bytes": "",
        "int8_state_bytes": "",
        "int4_state_bytes": "",
        "scale_bytes": "",
        "lowrank_factor_bytes": "",
        "residual_bytes": "",
        "compression_ratio": "",
        "memory_reduction_ratio": "",
        "max_sessions_full_state_under_1GB": "",
        "max_sessions_compressed_under_1GB": "",
        "session_capacity_gain": "",
        "checkpoint_compress_ms": "",
        "checkpoint_restore_ms": "",
        "transform_total_ms": "",
        "continuation_decode_ms": "",
        "reconstruction_overhead_ms": "",
        "priority_build_ms": "",
        "mask_build_ms": "",
        "lowrank_decompose_ms": "",
        "reconstruct_apply_ms": "",
        "quantize_ms": "",
        "decode_ms": "",
        "average_decode_latency_ms": "",
        "tokens_per_second": "",
        "mixed_selector_method": "",
        "mixed_selector_num_units": "",
        "mixed_selector_elapsed_ms": "",
        "mixed_selector_budget_abs_error": "",
        "mixed_selector_fp16_units": "",
        "mixed_selector_int8_units": "",
        "mixed_selector_int4_units": "",
        "mixed_selector_fp16_bytes": "",
        "mixed_selector_int8_bytes": "",
        "mixed_selector_int4_bytes": "",
        "budget_under_min": "",
        "budget_over_max": "",
        "clean_cache_changed": "",
        "alias_check_pass": "",
        "full_repeat_check": "",
        "status": status,
        "error_msg": error_msg,
    }


def make_result_row(
    args,
    run_id,
    sample,
    strategy,
    budget_ratio,
    effective_budget_ratio,
    full_loss,
    continuation_loss,
    full_ppl,
    continuation_ppl,
    logits_mse,
    top1_agreement,
    reconstruction_mse,
    original_state_elements,
    kept_state_elements,
    reconstructed_state_elements,
    original_state_bytes,
    compressed_state_bytes,
    estimated_state_memory_bytes,
    fp16_state_bytes,
    int8_state_bytes,
    int4_state_bytes,
    scale_bytes,
    lowrank_factor_bytes,
    residual_bytes,
    compression_ratio,
    memory_reduction_ratio,
    max_sessions_full_state_under_1GB,
    max_sessions_compressed_under_1GB,
    session_capacity_gain,
    checkpoint_compress_ms,
    checkpoint_restore_ms,
    transform_total_ms,
    continuation_decode_ms,
    reconstruction_overhead_ms,
    quantization_bits,
    full_precision_fraction,
    quantized_fraction,
    quantization_scale_mean,
    quantization_scale_max,
    quantization_mse,
    fallback_reason,
    priority_build_ms,
    mask_build_ms,
    lowrank_decompose_ms,
    reconstruct_apply_ms,
    quantize_ms,
    decode_ms,
    average_decode_latency_ms,
    tokens_per_second,
    clean_cache_changed,
    alias_check_pass,
    full_repeat_check,
    status,
    error_msg,
    mixed_selector_method="",
    mixed_selector_num_units="",
    mixed_selector_elapsed_ms="",
    mixed_selector_budget_abs_error="",
    mixed_selector_fp16_units="",
    mixed_selector_int8_units="",
    mixed_selector_int4_units="",
    mixed_selector_fp16_bytes="",
    mixed_selector_int8_bytes="",
    mixed_selector_int4_bytes="",
    budget_under_min="",
    budget_over_max="",
    hybrid_alpha="",
    rank="",
):
    strategy = STRATEGY_ALIASES.get(strategy, strategy)
    return {
        "run_id": run_id,
        "seed": getattr(args, "seed", ""),
        "dataset": sample["dataset"],
        "strategy": strategy,
        "reconstruction_method": RECONSTRUCTION_METHOD_MAP.get(strategy, ""),
        "rank": rank,
        "hybrid_alpha": hybrid_alpha,
        "budget_ratio": float(budget_ratio),
        "budget_mode": getattr(args, "budget_mode", ""),
        "effective_budget_ratio": effective_budget_ratio,
        "summary": PHASE24_SUMMARY_TEXT,
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
        "reconstruction_mse": reconstruction_mse,
        "quantization_bits": quantization_bits,
        "full_precision_fraction": full_precision_fraction,
        "quantized_fraction": quantized_fraction,
        "quantization_scale_mean": quantization_scale_mean,
        "quantization_scale_max": quantization_scale_max,
        "quantization_mse": quantization_mse,
        "fallback_reason": fallback_reason,
        "original_state_elements": original_state_elements,
        "kept_state_elements": kept_state_elements,
        "reconstructed_state_elements": reconstructed_state_elements,
        "original_state_bytes": original_state_bytes,
        "compressed_state_bytes": compressed_state_bytes,
        "estimated_state_memory_bytes": estimated_state_memory_bytes,
        "fp16_state_bytes": fp16_state_bytes,
        "int8_state_bytes": int8_state_bytes,
        "int4_state_bytes": int4_state_bytes,
        "scale_bytes": scale_bytes,
        "lowrank_factor_bytes": lowrank_factor_bytes,
        "residual_bytes": residual_bytes,
        "compression_ratio": compression_ratio,
        "memory_reduction_ratio": memory_reduction_ratio,
        "max_sessions_full_state_under_1GB": max_sessions_full_state_under_1GB,
        "max_sessions_compressed_under_1GB": max_sessions_compressed_under_1GB,
        "session_capacity_gain": session_capacity_gain,
        "checkpoint_compress_ms": checkpoint_compress_ms,
        "checkpoint_restore_ms": checkpoint_restore_ms,
        "transform_total_ms": transform_total_ms,
        "continuation_decode_ms": continuation_decode_ms,
        "reconstruction_overhead_ms": reconstruction_overhead_ms,
        "priority_build_ms": priority_build_ms,
        "mask_build_ms": mask_build_ms,
        "lowrank_decompose_ms": lowrank_decompose_ms,
        "reconstruct_apply_ms": reconstruct_apply_ms,
        "quantize_ms": quantize_ms,
        "decode_ms": decode_ms,
        "average_decode_latency_ms": average_decode_latency_ms,
        "tokens_per_second": tokens_per_second,
        "mixed_selector_method": mixed_selector_method,
        "mixed_selector_num_units": mixed_selector_num_units,
        "mixed_selector_elapsed_ms": mixed_selector_elapsed_ms,
        "mixed_selector_budget_abs_error": mixed_selector_budget_abs_error,
        "mixed_selector_fp16_units": mixed_selector_fp16_units,
        "mixed_selector_int8_units": mixed_selector_int8_units,
        "mixed_selector_int4_units": mixed_selector_int4_units,
        "mixed_selector_fp16_bytes": mixed_selector_fp16_bytes,
        "mixed_selector_int8_bytes": mixed_selector_int8_bytes,
        "mixed_selector_int4_bytes": mixed_selector_int4_bytes,
        "budget_under_min": budget_under_min,
        "budget_over_max": budget_over_max,
        "clean_cache_changed": clean_cache_changed,
        "alias_check_pass": alias_check_pass,
        "full_repeat_check": full_repeat_check,
        "status": status,
        "error_msg": error_msg,
    }


def _unit_identity(unit):
    return (
        int(unit["layer_idx"]),
        int(unit["head_id"]),
        int(unit["channel_group_id"]),
        int(unit["ssm_channel_index"]),
    )


def _group_key_head(unit):
    return int(unit["layer_idx"]), int(unit["head_id"])


def _group_key_layer(unit):
    return int(unit["layer_idx"])


def _get_cache_pair(cache, unit):
    return cache["key_value_memory_dict"][int(unit["layer_idx"])]


def get_state_tensor_bytes(tensor):
    return int(tensor.element_size() * tensor.numel()) if torch.is_tensor(tensor) else 0


def get_unit_parts(cache, unit):
    conv_state, ssm_state = _get_cache_pair(cache, unit)
    conv_idx = int(unit["conv_index"])
    head_id = int(unit["head_id"])
    channel_offset = int(unit["ssm_channel_index"])
    conv_part = conv_state[:, conv_idx, :]
    ssm_part = ssm_state[:, head_id, channel_offset, :]
    return conv_part, ssm_part


def get_unit_vector(cache, unit):
    conv_part, ssm_part = get_unit_parts(cache, unit)
    return torch.cat([conv_part.reshape(-1), ssm_part.reshape(-1)], dim=0)


def set_unit_vector(cache, unit, vector):
    conv_state, ssm_state = _get_cache_pair(cache, unit)
    conv_idx = int(unit["conv_index"])
    head_id = int(unit["head_id"])
    channel_offset = int(unit["ssm_channel_index"])
    conv_ref = conv_state[:, conv_idx, :]
    ssm_ref = ssm_state[:, head_id, channel_offset, :]
    conv_elems = int(conv_ref.numel())
    conv_part = vector[:conv_elems].reshape_as(conv_ref).to(device=conv_ref.device, dtype=conv_ref.dtype)
    ssm_part = vector[conv_elems:conv_elems + int(ssm_ref.numel())].reshape_as(ssm_ref).to(device=ssm_ref.device, dtype=ssm_ref.dtype)
    conv_state[:, conv_idx, :] = conv_part
    ssm_state[:, head_id, channel_offset, :] = ssm_part


def zero_out_units(cache, dropped_units):
    for unit in dropped_units:
        conv_state, ssm_state = _get_cache_pair(cache, unit)
        conv_idx = int(unit["conv_index"])
        head_id = int(unit["head_id"])
        channel_offset = int(unit["ssm_channel_index"])
        conv_state[:, conv_idx, :] = 0
        ssm_state[:, head_id, channel_offset, :] = 0


def compute_reconstruction_mse(clean_cache, candidate_cache, units):
    if not units:
        return 0.0
    sq_error_sum = 0.0
    total_elements = 0
    for unit in units:
        clean_vec = get_unit_vector(clean_cache, unit).float()
        cand_vec = get_unit_vector(candidate_cache, unit).float()
        diff = cand_vec - clean_vec
        sq_error_sum += float((diff * diff).sum().item())
        total_elements += int(diff.numel())
    if total_elements <= 0:
        return 0.0
    return sq_error_sum / total_elements


def estimate_original_state_bytes(clean_cache, mixer_pairs):
    total = 0
    for layer_idx, _mixer in mixer_pairs:
        if layer_idx not in clean_cache["key_value_memory_dict"]:
            continue
        conv_state, ssm_state = clean_cache["key_value_memory_dict"][layer_idx]
        total += get_state_tensor_bytes(conv_state) + get_state_tensor_bytes(ssm_state)
    return int(total)


def get_primary_state_dtype_bytes(clean_cache):
    for layer_idx in clean_cache["key_value_memory_dict"]:
        conv_state, _ = clean_cache["key_value_memory_dict"][layer_idx]
        return max(int(conv_state.element_size()), 1)
    return 1


def get_strategy_scoring_name(strategy):
    strategy = STRATEGY_ALIASES.get(strategy, strategy)
    if strategy in RANDOM_PRIORITY_STRATEGIES:
        return "random_drop"
    if strategy in UNIFORM_PRIORITY_STRATEGIES:
        return "uniform_drop"
    if strategy in STATE_NORM_PRIORITY_STRATEGIES:
        return "state_norm_keep"
    if strategy in HYBRID_PRIORITY_STRATEGIES:
        return "hybrid_norm_head"
    if strategy == "full_state":
        return None
    raise ValueError(f"Unsupported strategy for scoring: {strategy}")


def get_strategy_quantization_bits(strategy):
    strategy = STRATEGY_ALIASES.get(strategy, strategy)
    if strategy in {"state_quant_int8", "hybrid_zero_drop_plus_int8", "lowrank_reconstruct_plus_int8", "residual_lowrank_plus_int8"}:
        return 8
    if strategy == "state_quant_int4":
        return 4
    return None


def get_quantized_storage_fraction(bits, full_precision_bits=16):
    return float(bits) / float(full_precision_bits)


def minimum_supported_budget_ratio_for_quantization(strategy):
    bits = get_strategy_quantization_bits(strategy)
    if bits is None:
        return None
    return get_quantized_storage_fraction(bits)


def has_effective_head_sensitivity(score_maps):
    head_zscores = score_maps.get("head_zscores", {}) if isinstance(score_maps, dict) else {}
    if not head_zscores:
        return False
    for value in head_zscores.values():
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value) and abs(value) > 1e-12:
            return True
    return False


def quantized_unit_storage_bytes(unit, bits, scale_bytes=4):
    num_elements = int(unit["num_elements"])
    return int((num_elements * int(bits) + 7) // 8 + scale_bytes)


def quantized_tensor_data_bytes(num_elements, bits):
    return int((int(num_elements) * int(bits) + 7) // 8)


def quantized_unit_data_bytes(unit, bits):
    return quantized_tensor_data_bytes(int(unit["num_elements"]), bits)


def quantized_unit_scale_bytes(units, per_tensor_scale_bytes=4):
    return int(len(units) * int(per_tensor_scale_bytes))


def unit_num_bytes_sum(units):
    return int(sum(int(unit["num_bytes"]) for unit in units))


def unit_num_elements_sum(units):
    return int(sum(int(unit["num_elements"]) for unit in units))


def build_storage_breakdown(
    *,
    original_state_bytes,
    fp16_state_bytes,
    int8_state_bytes,
    int4_state_bytes,
    scale_bytes,
    lowrank_factor_bytes,
    residual_bytes,
):
    compressed_state_bytes = int(
        fp16_state_bytes + int8_state_bytes + int4_state_bytes + scale_bytes + lowrank_factor_bytes + residual_bytes
    )
    compression_ratio = (compressed_state_bytes / original_state_bytes) if original_state_bytes > 0 else 0.0
    memory_reduction_ratio = 1.0 - compression_ratio
    max_sessions_full = int(ONE_GB_BYTES // original_state_bytes) if original_state_bytes > 0 else 0
    max_sessions_compressed = int(ONE_GB_BYTES // compressed_state_bytes) if compressed_state_bytes > 0 else 0
    session_capacity_gain = (max_sessions_compressed / max_sessions_full) if max_sessions_full > 0 else ""
    return {
        "original_state_bytes": int(original_state_bytes),
        "compressed_state_bytes": int(compressed_state_bytes),
        "estimated_state_memory_bytes": int(compressed_state_bytes),
        "fp16_state_bytes": int(fp16_state_bytes),
        "int8_state_bytes": int(int8_state_bytes),
        "int4_state_bytes": int(int4_state_bytes),
        "scale_bytes": int(scale_bytes),
        "lowrank_factor_bytes": int(lowrank_factor_bytes),
        "residual_bytes": int(residual_bytes),
        "compression_ratio": float(compression_ratio),
        "memory_reduction_ratio": float(memory_reduction_ratio),
        "effective_budget_ratio": float(compression_ratio),
        "max_sessions_full_state_under_1GB": int(max_sessions_full),
        "max_sessions_compressed_under_1GB": int(max_sessions_compressed),
        "session_capacity_gain": session_capacity_gain,
    }


def estimate_lowrank_storage_components(layer_info, dropped_counts, rank, dtype_bytes, factor_bits=None, scales_per_layer=0):
    if rank is None:
        return 0, 0
    factor_bytes = 0
    scale_bytes = 0
    for layer_idx, meta in layer_info.items():
        rows = int(dropped_counts.get(layer_idx, 0))
        cols = int(meta["vector_dim"])
        if rows <= 0 or cols <= 0:
            continue
        rank_eff = min(int(rank), rows, cols)
        if rank_eff <= 0:
            continue
        left_elements = int(rows * rank_eff)
        right_elements = int(rank_eff * cols)
        if factor_bits is None:
            factor_bytes += int((left_elements + right_elements) * dtype_bytes)
        else:
            factor_bytes += quantized_tensor_data_bytes(left_elements, factor_bits)
            factor_bytes += quantized_tensor_data_bytes(right_elements, factor_bits)
            scale_bytes += int(scales_per_layer * 4)
    return int(factor_bytes), int(scale_bytes)


def build_layer_unit_metadata(units):
    info = defaultdict(lambda: {"count": 0, "vector_dim": 0})
    for unit in units:
        layer_idx = int(unit["layer_idx"])
        info[layer_idx]["count"] += 1
        info[layer_idx]["vector_dim"] = max(info[layer_idx]["vector_dim"], int(unit["num_elements"]))
    return info


def estimate_lowrank_factor_bytes(layer_info, dropped_counts, rank, dtype_bytes):
    factor_bytes, _scale_bytes = estimate_lowrank_storage_components(layer_info, dropped_counts, rank, dtype_bytes)
    return int(factor_bytes)


def estimate_storage_bytes_for_selection(strategy, inventory, selected_units, dropped_units, dtype_bytes, rank=None):
    strategy = STRATEGY_ALIASES.get(strategy, strategy)
    base_bytes = int(inventory["unmanaged_bytes"])
    selected_fp_bytes = unit_num_bytes_sum(selected_units)
    quant_bits = get_strategy_quantization_bits(strategy)
    quant_selected_bytes = 0
    quant_selected_scale_bytes = 0
    quant_dropped_bytes = 0
    quant_dropped_scale_bytes = 0
    if strategy in KEPT_QUANTIZATION_STRATEGIES and quant_bits is not None:
        quant_selected_bytes = sum(quantized_unit_data_bytes(unit, quant_bits) for unit in selected_units)
        quant_selected_scale_bytes = quantized_unit_scale_bytes(selected_units)
    if strategy in MIXED_QUANTIZATION_STRATEGIES and quant_bits is not None:
        quant_dropped_bytes = sum(quantized_unit_data_bytes(unit, quant_bits) for unit in dropped_units)
        quant_dropped_scale_bytes = quantized_unit_scale_bytes(dropped_units)
    lowrank_bytes = 0
    lowrank_scale_bytes = 0
    if strategy in LOWRANK_STRATEGIES:
        layer_info = build_layer_unit_metadata(inventory["manageable_units"])
        dropped_counts = defaultdict(int)
        for unit in dropped_units:
            dropped_counts[int(unit["layer_idx"])] += 1
        factor_bits = 8 if strategy in LOWRANK_FACTOR_INT8_STRATEGIES else None
        lowrank_bytes, lowrank_scale_bytes = estimate_lowrank_storage_components(
            layer_info,
            dropped_counts,
            rank,
            dtype_bytes,
            factor_bits=factor_bits,
            scales_per_layer=(2 if factor_bits is not None else 0),
        )

    if strategy in PRIORITY_MIXED_STRATEGIES:
        raise ValueError("priority_mixed_quant requires specialized effective-storage selection")
    if strategy in RESIDUAL_CORRECTION_STRATEGIES:
        layer_info = build_layer_unit_metadata(inventory["manageable_units"])
        full_counts = {int(layer_idx): int(meta["count"]) for layer_idx, meta in layer_info.items()}
        residual_bytes, _ = estimate_lowrank_storage_components(layer_info, full_counts, rank, dtype_bytes)
        int8_bytes = sum(quantized_unit_data_bytes(unit, 8) for unit in inventory["manageable_units"])
        scale_bytes = quantized_unit_scale_bytes(inventory["manageable_units"])
        return int(base_bytes + int8_bytes + scale_bytes + residual_bytes)

    if strategy in MIXED_QUANTIZATION_STRATEGIES:
        return int(base_bytes + selected_fp_bytes + quant_dropped_bytes + quant_dropped_scale_bytes)
    if strategy in KEPT_QUANTIZATION_STRATEGIES:
        return int(base_bytes + quant_selected_bytes + quant_selected_scale_bytes + lowrank_bytes + lowrank_scale_bytes)
    if strategy in LOWRANK_FACTOR_INT8_STRATEGIES:
        return int(base_bytes + selected_fp_bytes + lowrank_bytes + lowrank_scale_bytes)
    if strategy in LOWRANK_STRATEGIES:
        return int(base_bytes + selected_fp_bytes + lowrank_bytes)
    return int(base_bytes + selected_fp_bytes)


def build_prefix_unit_stats(manageable_units, strategy):
    prefix_fp_bytes = [0]
    prefix_quant_bytes = [0]
    prefix_int8_bytes = [0]
    prefix_int4_bytes = [0]
    prefix_elements = [0]
    running_fp_bytes = 0
    running_quant_bytes = 0
    running_int8_bytes = 0
    running_int4_bytes = 0
    running_elements = 0
    quant_bits = get_strategy_quantization_bits(strategy)
    for unit in manageable_units:
        running_fp_bytes += int(unit["num_bytes"])
        if quant_bits is not None:
            running_quant_bytes += quantized_unit_storage_bytes(unit, quant_bits)
        running_int8_bytes += quantized_unit_data_bytes(unit, 8) + 4
        running_int4_bytes += quantized_unit_data_bytes(unit, 4) + 4
        running_elements += int(unit["num_elements"])
        prefix_fp_bytes.append(running_fp_bytes)
        prefix_quant_bytes.append(running_quant_bytes)
        prefix_int8_bytes.append(running_int8_bytes)
        prefix_int4_bytes.append(running_int4_bytes)
        prefix_elements.append(running_elements)
    return {
        "prefix_fp_bytes": prefix_fp_bytes,
        "prefix_quant_bytes": prefix_quant_bytes,
        "prefix_int8_bytes": prefix_int8_bytes,
        "prefix_int4_bytes": prefix_int4_bytes,
        "prefix_elements": prefix_elements,
        "total_fp_bytes": running_fp_bytes,
        "total_quant_bytes": running_quant_bytes,
        "total_int8_bytes": running_int8_bytes,
        "total_int4_bytes": running_int4_bytes,
        "total_elements": running_elements,
    }


def build_layer_count_prefixes(manageable_units, layer_info):
    layer_indices = sorted(int(layer_idx) for layer_idx in layer_info.keys())
    prefix_counts = {layer_idx: [0] for layer_idx in layer_indices}
    running_counts = {layer_idx: 0 for layer_idx in layer_indices}
    for unit in manageable_units:
        unit_layer_idx = int(unit["layer_idx"])
        running_counts[unit_layer_idx] += 1
        for layer_idx in layer_indices:
            prefix_counts[layer_idx].append(running_counts[layer_idx])
    return prefix_counts


def build_cost_prefix_sums(costs):
    prefix = [0]
    running = 0
    for cost in costs:
        running += int(cost)
        prefix.append(int(running))
    return prefix


def build_mixed_quant_selector_cache(sorted_units):
    fp16_costs = [int(unit["num_bytes"]) for unit in sorted_units]
    int8_costs = [quantized_unit_storage_bytes(unit, 8) for unit in sorted_units]
    int4_costs = [quantized_unit_storage_bytes(unit, 4) for unit in sorted_units]
    prefix_fp16 = build_cost_prefix_sums(fp16_costs)
    prefix_int8 = build_cost_prefix_sums(int8_costs)
    prefix_int4 = build_cost_prefix_sums(int4_costs)
    prefix_fp16_minus_int8 = [int(a - b) for a, b in zip(prefix_fp16, prefix_int8, strict=False)]
    prefix_int8_minus_int4 = [int(a - b) for a, b in zip(prefix_int8, prefix_int4, strict=False)]
    return {
        "sorted_units": list(sorted_units),
        "fp16_costs": fp16_costs,
        "int8_costs": int8_costs,
        "int4_costs": int4_costs,
        "prefix_fp16": prefix_fp16,
        "prefix_int8": prefix_int8,
        "prefix_int4": prefix_int4,
        "prefix_fp16_minus_int8": prefix_fp16_minus_int8,
        "prefix_int8_minus_int4": prefix_int8_minus_int4,
        "total_fp16_bytes": int(prefix_fp16[-1]),
        "total_int8_bytes": int(prefix_int8[-1]),
        "total_int4_bytes": int(prefix_int4[-1]),
        "num_units": int(len(sorted_units)),
    }


def _is_better_storage_candidate(candidate_storage, candidate_error, best_storage, best_error, target_bytes):
    if best_storage is None or best_error is None:
        return True
    if candidate_error < best_error:
        return True
    if candidate_error > best_error:
        return False
    if candidate_storage <= target_bytes and best_storage > target_bytes:
        return True
    return False


def _build_mixed_selector_result(selector_cache, fp_end, int8_end, target_bytes, *, method, elapsed_ms=0.0, budget_under_min=False, budget_over_max=False, error_msg=""):
    prefix_fp16 = selector_cache["prefix_fp16"]
    prefix_int8 = selector_cache["prefix_int8"]
    prefix_int4 = selector_cache["prefix_int4"]
    total_int4_bytes = int(selector_cache["total_int4_bytes"])
    fp16_bytes = int(prefix_fp16[fp_end])
    int8_bytes = int(prefix_int8[int8_end] - prefix_int8[fp_end])
    int4_bytes = int(total_int4_bytes - prefix_int4[int8_end])
    selected_bytes = int(fp16_bytes + int8_bytes + int4_bytes)
    return {
        "fp_end": int(fp_end),
        "int8_end": int(int8_end),
        "selected_bytes": int(selected_bytes),
        "budget_abs_error": float(abs(selected_bytes - target_bytes)),
        "fp16_units": int(fp_end),
        "int8_units": int(max(0, int8_end - fp_end)),
        "int4_units": int(max(0, selector_cache["num_units"] - int8_end)),
        "fp16_bytes": int(fp16_bytes),
        "int8_bytes": int(int8_bytes),
        "int4_bytes": int(int4_bytes),
        "method": method,
        "num_units": int(selector_cache["num_units"]),
        "elapsed_ms": float(elapsed_ms),
        "budget_under_min": bool(budget_under_min),
        "budget_over_max": bool(budget_over_max),
        "status": "error" if error_msg else "ok",
        "error_msg": error_msg,
    }


def choose_mixed_quant_thresholds_exhaustive(sorted_units, target_bytes, fp16_costs, int8_costs, int4_costs):
    selector_cache = build_mixed_quant_selector_cache(sorted_units)
    num_units = int(selector_cache["num_units"])
    if num_units <= 0:
        return {
            "fp_end": 0,
            "int8_end": 0,
            "selected_bytes": 0,
            "budget_abs_error": float(abs(target_bytes)),
            "status": "error",
            "error_msg": "priority_mixed_quant selector received zero manageable units",
        }
    total_int4_bytes = int(sum(int(cost) for cost in int4_costs))
    total_fp16_bytes = int(sum(int(cost) for cost in fp16_costs))
    if target_bytes <= total_int4_bytes:
        return _build_mixed_selector_result(selector_cache, 0, 0, target_bytes, method="exhaustive_reference", budget_under_min=True)
    if target_bytes >= total_fp16_bytes:
        return _build_mixed_selector_result(selector_cache, num_units, num_units, target_bytes, method="exhaustive_reference", budget_over_max=True)

    prefix_fp16 = selector_cache["prefix_fp16"]
    prefix_int8 = selector_cache["prefix_int8"]
    prefix_int4 = selector_cache["prefix_int4"]
    total_int4_bytes = int(selector_cache["total_int4_bytes"])
    best_fp_end = 0
    best_int8_end = 0
    best_storage = None
    best_error = None
    for int8_end in range(0, num_units + 1):
        suffix_int4_bytes = int(total_int4_bytes - prefix_int4[int8_end])
        for fp_end in range(0, int8_end + 1):
            storage_bytes = int(prefix_fp16[fp_end] + (prefix_int8[int8_end] - prefix_int8[fp_end]) + suffix_int4_bytes)
            abs_error = float(abs(storage_bytes - target_bytes))
            if _is_better_storage_candidate(storage_bytes, abs_error, best_storage, best_error, target_bytes):
                best_fp_end = int(fp_end)
                best_int8_end = int(int8_end)
                best_storage = int(storage_bytes)
                best_error = float(abs_error)
    return _build_mixed_selector_result(selector_cache, best_fp_end, best_int8_end, target_bytes, method="exhaustive_reference")


def choose_mixed_quant_thresholds_fast(sorted_units, target_bytes, fp16_costs, int8_costs, int4_costs, selector_cache=None):
    started = time.perf_counter()
    if selector_cache is None:
        selector_cache = build_mixed_quant_selector_cache(sorted_units)
    num_units = int(selector_cache["num_units"])
    if num_units <= 0:
        return {
            "fp_end": 0,
            "int8_end": 0,
            "selected_bytes": 0,
            "budget_abs_error": float(abs(target_bytes)),
            "fp16_units": 0,
            "int8_units": 0,
            "int4_units": 0,
            "fp16_bytes": 0,
            "int8_bytes": 0,
            "int4_bytes": 0,
            "method": MIXED_SELECTOR_METHOD,
            "num_units": 0,
            "elapsed_ms": float((time.perf_counter() - started) * 1000.0),
            "budget_under_min": False,
            "budget_over_max": False,
            "status": "error",
            "error_msg": "priority_mixed_quant selector received zero manageable units",
        }

    total_int4_bytes = int(selector_cache["total_int4_bytes"])
    total_fp16_bytes = int(selector_cache["total_fp16_bytes"])
    if target_bytes <= total_int4_bytes:
        return _build_mixed_selector_result(
            selector_cache,
            0,
            0,
            target_bytes,
            method=MIXED_SELECTOR_METHOD,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            budget_under_min=True,
        )
    if target_bytes >= total_fp16_bytes:
        return _build_mixed_selector_result(
            selector_cache,
            num_units,
            num_units,
            target_bytes,
            method=MIXED_SELECTOR_METHOD,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            budget_over_max=True,
        )

    prefix_fp16_minus_int8 = selector_cache["prefix_fp16_minus_int8"]
    prefix_int8_minus_int4 = selector_cache["prefix_int8_minus_int4"]
    best_fp_end = 0
    best_int8_end = 0
    best_storage = None
    best_error = None
    base_int4_bytes = int(total_int4_bytes)
    for fp_end in range(0, num_units + 1):
        required_b = float(target_bytes - base_int4_bytes - prefix_fp16_minus_int8[fp_end])
        pos = bisect.bisect_left(prefix_int8_minus_int4, required_b, lo=fp_end)
        candidates = []
        if fp_end <= num_units:
            candidates.append(fp_end)
        if pos <= num_units:
            candidates.append(pos)
        if pos - 1 >= fp_end:
            candidates.append(pos - 1)
        if num_units >= fp_end:
            candidates.append(num_units)
        for int8_end in sorted(set(candidates)):
            storage_bytes = int(base_int4_bytes + prefix_fp16_minus_int8[fp_end] + prefix_int8_minus_int4[int8_end])
            abs_error = float(abs(storage_bytes - target_bytes))
            if _is_better_storage_candidate(storage_bytes, abs_error, best_storage, best_error, target_bytes):
                best_fp_end = int(fp_end)
                best_int8_end = int(int8_end)
                best_storage = int(storage_bytes)
                best_error = float(abs_error)
    return _build_mixed_selector_result(
        selector_cache,
        best_fp_end,
        best_int8_end,
        target_bytes,
        method=MIXED_SELECTOR_METHOD,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def validate_mixed_quant_selector_small(selector_cache, target_bytes, validate_k):
    subset_n = int(min(max(int(validate_k), 0), int(selector_cache.get("num_units", 0))))
    if subset_n <= 0:
        return ""
    subset_units = selector_cache["sorted_units"][:subset_n]
    subset_fp16 = selector_cache["fp16_costs"][:subset_n]
    subset_int8 = selector_cache["int8_costs"][:subset_n]
    subset_int4 = selector_cache["int4_costs"][:subset_n]
    subset_cache = build_mixed_quant_selector_cache(subset_units)
    fast = choose_mixed_quant_thresholds_fast(subset_units, target_bytes, subset_fp16, subset_int8, subset_int4, selector_cache=subset_cache)
    exhaustive = choose_mixed_quant_thresholds_exhaustive(subset_units, target_bytes, subset_fp16, subset_int8, subset_int4)
    selected_diff = abs(int(fast["selected_bytes"]) - int(exhaustive["selected_bytes"]))
    error_diff = abs(float(fast["budget_abs_error"]) - float(exhaustive["budget_abs_error"]))
    if selected_diff > 1 or error_diff > 1e-6:
        return (
            "mixed_quant_selector_validation_warning: "
            f"k={subset_n} fast_bytes={int(fast['selected_bytes'])} exhaustive_bytes={int(exhaustive['selected_bytes'])} "
            f"fast_error={float(fast['budget_abs_error']):.6f} exhaustive_error={float(exhaustive['budget_abs_error']):.6f}"
        )
    return ""


def compute_lowrank_storage_from_kept_counts(layer_info, prefix_layer_counts, idx, rank, dtype_bytes):
    if rank is None:
        return 0, 0
    factor_bytes = 0
    scale_bytes = 0
    rank_value = int(rank)
    for layer_idx, meta in layer_info.items():
        total_count = int(meta["count"])
        kept_count = int(prefix_layer_counts[int(layer_idx)][idx])
        dropped_count = max(0, total_count - kept_count)
        cols = int(meta["vector_dim"])
        if dropped_count <= 0 or cols <= 0:
            continue
        rank_eff = min(rank_value, dropped_count, cols)
        if rank_eff <= 0:
            continue
        left_elements = int(dropped_count * rank_eff)
        right_elements = int(rank_eff * cols)
        factor_bytes += int((left_elements + right_elements) * dtype_bytes)
    return int(factor_bytes), int(scale_bytes)


def select_units_for_effective_storage(scored_units, inventory, strategy, target_budget_ratio, dtype_bytes, rank=None, mixed_selector_cache=None, validate_mixed_quant_selector=False, mixed_quant_validate_k=128):
    strategy = STRATEGY_ALIASES.get(strategy, strategy)
    total_cache_bytes = int(inventory["total_cache_bytes"])
    target_total_bytes = float(target_budget_ratio) * total_cache_bytes
    manageable_units = list(scored_units)
    layer_info = build_layer_unit_metadata(manageable_units)
    prefix_stats = build_prefix_unit_stats(manageable_units, strategy)
    prefix_layer_counts = build_layer_count_prefixes(manageable_units, layer_info) if strategy in LOWRANK_STRATEGIES else None

    base_bytes = int(inventory["unmanaged_bytes"])
    total_quant_bytes = int(prefix_stats["total_quant_bytes"])

    if strategy in PRIORITY_MIXED_STRATEGIES:
        if mixed_selector_cache is None:
            mixed_selector_cache = build_mixed_quant_selector_cache(manageable_units)
        target_manageable_bytes = float(target_total_bytes - base_bytes)
        mixed_selector_result = choose_mixed_quant_thresholds_fast(
            manageable_units,
            target_manageable_bytes,
            mixed_selector_cache["fp16_costs"],
            mixed_selector_cache["int8_costs"],
            mixed_selector_cache["int4_costs"],
            selector_cache=mixed_selector_cache,
        )
        if mixed_selector_result.get("status") == "error":
            raise RuntimeError(f"status=error: {mixed_selector_result.get('error_msg', 'mixed quant selector failed')}")
        validation_warning = ""
        if validate_mixed_quant_selector:
            validation_warning = validate_mixed_quant_selector_small(mixed_selector_cache, target_manageable_bytes, mixed_quant_validate_k)
            if validation_warning:
                print(f"[WARN] {validation_warning}")
        best_fp_idx = int(mixed_selector_result["fp_end"])
        best_int8_idx = int(mixed_selector_result["int8_end"])
        best_storage_bytes = int(base_bytes + mixed_selector_result["selected_bytes"])
        best_abs_error = float(abs(best_storage_bytes - target_total_bytes))
        selected_units = manageable_units[:best_fp_idx]
        int8_units = manageable_units[best_fp_idx:best_int8_idx]
        int4_units = manageable_units[best_int8_idx:]
        kept_manageable_elements = int(prefix_stats["prefix_elements"][best_fp_idx])
        effective_budget_ratio = (best_storage_bytes / total_cache_bytes) if total_cache_bytes > 0 else 0.0
        return {
            "selected_units": selected_units,
            "dropped_units": manageable_units[best_fp_idx:],
            "mixed_fp16_units": selected_units,
            "mixed_int8_units": int8_units,
            "mixed_int4_units": int4_units,
            "kept_manageable_elements": kept_manageable_elements,
            "effective_budget_ratio": effective_budget_ratio,
            "budget_abs_error": best_abs_error,
            "estimated_state_memory_bytes": int(best_storage_bytes),
            "mixed_selector_method": mixed_selector_result.get("method", MIXED_SELECTOR_METHOD),
            "mixed_selector_num_units": int(mixed_selector_result.get("num_units", len(manageable_units))),
            "mixed_selector_elapsed_ms": float(mixed_selector_result.get("elapsed_ms", 0.0)),
            "mixed_selector_budget_abs_error": float(best_abs_error),
            "mixed_selector_fp16_units": int(mixed_selector_result.get("fp16_units", len(selected_units))),
            "mixed_selector_int8_units": int(mixed_selector_result.get("int8_units", len(int8_units))),
            "mixed_selector_int4_units": int(mixed_selector_result.get("int4_units", len(int4_units))),
            "mixed_selector_fp16_bytes": int(mixed_selector_result.get("fp16_bytes", 0)),
            "mixed_selector_int8_bytes": int(mixed_selector_result.get("int8_bytes", 0)),
            "mixed_selector_int4_bytes": int(mixed_selector_result.get("int4_bytes", 0)),
            "budget_under_min": bool(mixed_selector_result.get("budget_under_min", False)),
            "budget_over_max": bool(mixed_selector_result.get("budget_over_max", False)),
            "validation_warning": validation_warning,
        }

    def storage_bytes_for_idx(idx):
        selected_fp_bytes = int(prefix_stats["prefix_fp_bytes"][idx])
        selected_quant_bytes = int(prefix_stats["prefix_quant_bytes"][idx])
        dropped_quant_bytes = int(total_quant_bytes - selected_quant_bytes)
        lowrank_bytes = 0
        lowrank_scale_bytes = 0
        if strategy in LOWRANK_STRATEGIES:
            dropped_counts = {}
            for layer_idx, meta in layer_info.items():
                total_count = int(meta["count"])
                kept_count = int(prefix_layer_counts[int(layer_idx)][idx])
                dropped_counts[int(layer_idx)] = max(0, total_count - kept_count)
            factor_bits = 8 if strategy in LOWRANK_FACTOR_INT8_STRATEGIES else None
            lowrank_bytes, lowrank_scale_bytes = estimate_lowrank_storage_components(
                layer_info,
                dropped_counts,
                rank,
                dtype_bytes,
                factor_bits=factor_bits,
                scales_per_layer=(2 if factor_bits is not None else 0),
            )
        if strategy in MIXED_QUANTIZATION_STRATEGIES:
            return int(base_bytes + selected_fp_bytes + dropped_quant_bytes)
        if strategy in KEPT_QUANTIZATION_STRATEGIES:
            return int(base_bytes + selected_quant_bytes + lowrank_bytes)
        if strategy in LOWRANK_FACTOR_INT8_STRATEGIES:
            return int(base_bytes + selected_fp_bytes + lowrank_bytes + lowrank_scale_bytes)
        if strategy in LOWRANK_STRATEGIES:
            return int(base_bytes + selected_fp_bytes + lowrank_bytes)
        return int(base_bytes + selected_fp_bytes)

    best_idx = 0
    best_storage_bytes = storage_bytes_for_idx(0)
    best_abs_error = abs(best_storage_bytes - target_total_bytes)
    for idx in range(1, len(manageable_units) + 1):
        storage_bytes = storage_bytes_for_idx(idx)
        abs_error = abs(storage_bytes - target_total_bytes)
        if abs_error < best_abs_error or (abs_error == best_abs_error and storage_bytes <= target_total_bytes):
            best_idx = idx
            best_storage_bytes = storage_bytes
            best_abs_error = abs_error

    selected_units = manageable_units[:best_idx]
    kept_manageable_elements = int(prefix_stats["prefix_elements"][best_idx])
    effective_budget_ratio = (best_storage_bytes / total_cache_bytes) if total_cache_bytes > 0 else 0.0
    return {
        "selected_units": selected_units,
        "dropped_units": manageable_units[best_idx:],
        "kept_manageable_elements": kept_manageable_elements,
        "effective_budget_ratio": effective_budget_ratio,
        "budget_abs_error": best_abs_error,
        "estimated_state_memory_bytes": int(best_storage_bytes),
    }


def symmetric_quantize_vector(vector, bits):
    vector_fp = vector.float()
    if vector_fp.numel() == 0:
        return vector_fp, 0.0, 0.0
    qmax = (1 << (int(bits) - 1)) - 1
    if qmax <= 0:
        return vector_fp.clone(), 0.0, 0.0
    max_abs = float(vector_fp.abs().max().item())
    if max_abs <= 0.0:
        return torch.zeros_like(vector_fp), 0.0, 0.0
    scale = max_abs / qmax
    quantized = torch.clamp(torch.round(vector_fp / scale), -qmax, qmax)
    dequantized = quantized * scale
    mse = float(((dequantized - vector_fp) ** 2).mean().item())
    return dequantized, mse, float(scale)


def symmetric_quantize_tensor_to_codes(tensor, bits):
    tensor_fp = tensor.float()
    if tensor_fp.numel() == 0:
        return torch.zeros_like(tensor_fp, dtype=torch.int8), 0.0, tensor_fp.clone(), 0.0
    qmax = (1 << (int(bits) - 1)) - 1
    if qmax <= 0:
        return torch.zeros_like(tensor_fp, dtype=torch.int8), 0.0, tensor_fp.clone(), 0.0
    max_abs = float(tensor_fp.abs().max().item())
    if max_abs <= 0.0:
        zeros = torch.zeros_like(tensor_fp, dtype=torch.int8)
        return zeros, 0.0, torch.zeros_like(tensor_fp), 0.0
    scale = max_abs / qmax
    quantized = torch.clamp(torch.round(tensor_fp / scale), -qmax, qmax).to(torch.int8)
    dequantized = quantized.float() * scale
    mse = float(((dequantized - tensor_fp) ** 2).mean().item())
    return quantized, float(scale), dequantized, mse


def restore_quantized_tensor_from_codes(quantized, scale):
    return quantized.float() * float(scale)


def build_quantized_unit_payload(clean_cache, units, bits):
    entries = []
    total_sq_error = 0.0
    total_elements = 0
    scales = []
    sample_shape = ()
    sample_dtype = ""
    sample_device = ""
    for unit in units:
        clean_vec = get_unit_vector(clean_cache, unit)
        if not sample_shape:
            sample_shape = tuple(clean_vec.shape)
            sample_dtype = str(clean_vec.dtype)
            sample_device = str(clean_vec.device)
        quantized, scale, dequantized, mse = symmetric_quantize_tensor_to_codes(clean_vec, bits)
        entries.append({"unit": unit, "quantized": quantized, "scale": float(scale)})
        scales.append(float(scale))
        total_sq_error += float(mse * clean_vec.numel())
        total_elements += int(clean_vec.numel())
    scale_mean = (sum(scales) / len(scales)) if scales else 0.0
    scale_max = max(scales) if scales else 0.0
    return {
        "entries": entries,
        "quantization_mse": (total_sq_error / total_elements) if total_elements > 0 else 0.0,
        "num_quantized_elements": int(total_elements),
        "quantization_scale_mean": float(scale_mean),
        "quantization_scale_max": float(scale_max),
        "sample_shape": sample_shape,
        "sample_dtype": sample_dtype,
        "sample_device": sample_device,
        "data_bytes": sum(quantized_unit_data_bytes(unit, bits) for unit in units),
        "scale_bytes": quantized_unit_scale_bytes(units),
    }


def restore_quantized_unit_payload(modified_cache, payload_entries):
    restored_elements = 0
    for entry in payload_entries:
        restored_vec = restore_quantized_tensor_from_codes(entry["quantized"], entry["scale"])
        set_unit_vector(modified_cache, entry["unit"], restored_vec)
        restored_elements += int(restored_vec.numel())
    return int(restored_elements)


def build_mean_checkpoint_payload(clean_cache, kept_units, dropped_units, use_scale):
    kept_by_head = defaultdict(list)
    kept_by_layer = defaultdict(list)
    for unit in kept_units:
        kept_by_head[_group_key_head(unit)].append(unit)
        kept_by_layer[_group_key_layer(unit)].append(unit)
    global_kept = list(kept_units)
    stats_cache = {}
    stored_elements = 0

    def resolve_group_units(drop_unit):
        head_key = _group_key_head(drop_unit)
        if kept_by_head.get(head_key):
            return ("head", head_key), kept_by_head[head_key]
        layer_key = _group_key_layer(drop_unit)
        if kept_by_layer.get(layer_key):
            return ("layer", layer_key), kept_by_layer[layer_key]
        return ("global", "all"), global_kept

    for unit in dropped_units:
        stat_key, source_units = resolve_group_units(unit)
        if stat_key in stats_cache:
            continue
        if not source_units:
            stats_cache[stat_key] = None
            continue
        matrix = _stack_vectors(clean_cache, source_units)
        mean_vec = matrix.mean(dim=0)
        if use_scale:
            std_vec = matrix.std(dim=0, unbiased=False) if matrix.shape[0] > 1 else torch.zeros_like(mean_vec)
            prototype = mean_vec + torch.sign(mean_vec - mean_vec.mean()) * std_vec
            stats_cache[stat_key] = prototype.to(dtype=get_unit_vector(clean_cache, source_units[0]).dtype)
            stored_elements += int(mean_vec.numel() + std_vec.numel())
        else:
            stats_cache[stat_key] = mean_vec.to(dtype=get_unit_vector(clean_cache, source_units[0]).dtype)
            stored_elements += int(mean_vec.numel())
    assignments = []
    for unit in dropped_units:
        stat_key, _source_units = resolve_group_units(unit)
        assignments.append((unit, stats_cache.get(stat_key)))
    return {"assignments": assignments, "stored_elements": int(stored_elements)}


def restore_mean_checkpoint_payload(modified_cache, payload):
    reconstructed_elements = 0
    for unit, prototype in payload.get("assignments", []):
        if prototype is None:
            recon_vec = torch.zeros_like(get_unit_vector(modified_cache, unit).float())
        else:
            recon_vec = prototype
        set_unit_vector(modified_cache, unit, recon_vec)
        reconstructed_elements += int(unit["num_elements"])
    return int(reconstructed_elements)


def build_lowrank_checkpoint_payload(clean_cache, dropped_units, rank, factor_bits=None):
    dropped_by_layer = defaultdict(list)
    for unit in dropped_units:
        dropped_by_layer[_group_key_layer(unit)].append(unit)

    payload_layers = []
    factor_bytes = 0
    scale_bytes = 0
    reconstructed_elements = 0
    lowrank_decompose_ms = 0.0
    quantization_sq_error = 0.0
    quantization_elements = 0
    quantization_scales = []
    base_dtype_bytes = get_primary_state_dtype_bytes(clean_cache)

    for layer_idx, units in dropped_by_layer.items():
        if not units:
            continue
        matrix = _stack_vectors(clean_cache, units)
        if matrix is None or matrix.numel() == 0:
            continue
        rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
        rank_eff = min(int(rank or 0), rows, cols)
        if rank_eff <= 0:
            continue
        decompose_started = time.perf_counter()
        try:
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        except Exception:
            u, s, vh = torch.linalg.svd(matrix.cpu(), full_matrices=False)
            u = u.to(matrix.device)
            s = s.to(matrix.device)
            vh = vh.to(matrix.device)
        lowrank_decompose_ms += (time.perf_counter() - decompose_started) * 1000.0
        left = (u[:, :rank_eff] * s[:rank_eff]).contiguous()
        right = vh[:rank_eff, :].contiguous()
        layer_payload = {"layer_idx": layer_idx, "units": units, "rank_eff": rank_eff}
        if factor_bits is None:
            ref_dtype = get_unit_vector(clean_cache, units[0]).dtype
            left = left.to(dtype=ref_dtype)
            right = right.to(dtype=ref_dtype)
            layer_payload.update({"left": left, "right": right, "factor_bits": None})
            factor_bytes += int((left.numel() + right.numel()) * base_dtype_bytes)
        else:
            left_q, left_scale, left_deq, left_mse = symmetric_quantize_tensor_to_codes(left, factor_bits)
            right_q, right_scale, right_deq, right_mse = symmetric_quantize_tensor_to_codes(right, factor_bits)
            layer_payload.update(
                {
                    "left_q": left_q,
                    "left_scale": float(left_scale),
                    "right_q": right_q,
                    "right_scale": float(right_scale),
                    "factor_bits": int(factor_bits),
                }
            )
            factor_bytes += quantized_tensor_data_bytes(left_q.numel(), factor_bits)
            factor_bytes += quantized_tensor_data_bytes(right_q.numel(), factor_bits)
            scale_bytes += 8
            quantization_sq_error += float(left_mse * left.numel() + right_mse * right.numel())
            quantization_elements += int(left.numel() + right.numel())
            quantization_scales.extend([float(left_scale), float(right_scale)])
        reconstructed_elements += int(sum(int(unit["num_elements"]) for unit in units))
        payload_layers.append(layer_payload)
    return {
        "layers": payload_layers,
        "lowrank_factor_bytes": int(factor_bytes),
        "scale_bytes": int(scale_bytes),
        "reconstructed_state_elements": int(reconstructed_elements),
        "lowrank_decompose_ms": float(lowrank_decompose_ms),
        "quantization_mse": (quantization_sq_error / quantization_elements) if quantization_elements > 0 else "",
        "quantization_scale_mean": (sum(quantization_scales) / len(quantization_scales)) if quantization_scales else "",
        "quantization_scale_max": max(quantization_scales) if quantization_scales else "",
    }


def restore_lowrank_checkpoint_payload(modified_cache, payload):
    reconstructed_elements = 0
    for layer_payload in payload.get("layers", []):
        if layer_payload.get("factor_bits") is None:
            left = layer_payload["left"].float()
            right = layer_payload["right"].float()
        else:
            left = restore_quantized_tensor_from_codes(layer_payload["left_q"], layer_payload["left_scale"])
            right = restore_quantized_tensor_from_codes(layer_payload["right_q"], layer_payload["right_scale"])
        approx = left @ right
        for unit, row in zip(layer_payload["units"], approx, strict=False):
            set_unit_vector(modified_cache, unit, row)
            reconstructed_elements += int(unit["num_elements"])
    return int(reconstructed_elements)


def build_residual_lowrank_plus_int8_payload(clean_cache, units, rank):
    base_quant_payload = build_quantized_unit_payload(clean_cache, units, 8)
    units_by_layer = defaultdict(list)
    for entry in base_quant_payload["entries"]:
        units_by_layer[int(entry["unit"]["layer_idx"])].append(entry)

    residual_layers = []
    residual_bytes = 0
    lowrank_decompose_ms = 0.0
    dtype_bytes = get_primary_state_dtype_bytes(clean_cache)
    for layer_idx, entries in units_by_layer.items():
        clean_rows = []
        base_rows = []
        units_for_layer = []
        for entry in entries:
            unit = entry["unit"]
            units_for_layer.append(unit)
            clean_rows.append(get_unit_vector(clean_cache, unit).float())
            base_rows.append(restore_quantized_tensor_from_codes(entry["quantized"], entry["scale"]))
        clean_matrix = torch.stack(clean_rows, dim=0)
        base_matrix = torch.stack(base_rows, dim=0)
        residual_matrix = clean_matrix - base_matrix
        rows, cols = int(residual_matrix.shape[0]), int(residual_matrix.shape[1])
        rank_eff = min(int(rank or 0), rows, cols)
        if rank_eff <= 0:
            continue
        decompose_started = time.perf_counter()
        try:
            u, s, vh = torch.linalg.svd(residual_matrix, full_matrices=False)
        except Exception:
            u, s, vh = torch.linalg.svd(residual_matrix.cpu(), full_matrices=False)
            u = u.to(residual_matrix.device)
            s = s.to(residual_matrix.device)
            vh = vh.to(residual_matrix.device)
        lowrank_decompose_ms += (time.perf_counter() - decompose_started) * 1000.0
        ref_dtype = get_unit_vector(clean_cache, units_for_layer[0]).dtype
        left = (u[:, :rank_eff] * s[:rank_eff]).to(dtype=ref_dtype).contiguous()
        right = vh[:rank_eff, :].to(dtype=ref_dtype).contiguous()
        residual_bytes += int((left.numel() + right.numel()) * dtype_bytes)
        residual_layers.append({"units": units_for_layer, "left": left, "right": right})

    return {
        "base_quant_payload": base_quant_payload,
        "residual_layers": residual_layers,
        "residual_bytes": int(residual_bytes),
        "lowrank_decompose_ms": float(lowrank_decompose_ms),
    }


def restore_residual_lowrank_plus_int8_payload(modified_cache, payload):
    reconstructed_elements = restore_quantized_unit_payload(modified_cache, payload.get("base_quant_payload", {}).get("entries", []))
    for layer_payload in payload.get("residual_layers", []):
        correction = layer_payload["left"].float() @ layer_payload["right"].float()
        for unit, row in zip(layer_payload["units"], correction, strict=False):
            current = get_unit_vector(modified_cache, unit).float()
            set_unit_vector(modified_cache, unit, current + row)
    return int(reconstructed_elements)


def apply_quantization_to_units(clean_cache, modified_cache, units, bits):
    if not units:
        return {
            "quantization_mse": 0.0,
            "num_quantized_elements": 0,
            "quantization_scale_mean": 0.0,
            "quantization_scale_max": 0.0,
            "sample_shape": (),
            "sample_dtype": "",
            "sample_device": "",
        }
    total_sq_error = 0.0
    total_elements = 0
    scales = []
    sample_shape = ()
    sample_dtype = ""
    sample_device = ""
    for unit in units:
        clean_vec = get_unit_vector(clean_cache, unit)
        if not sample_shape:
            sample_shape = tuple(clean_vec.shape)
            sample_dtype = str(clean_vec.dtype)
            sample_device = str(clean_vec.device)
        dequantized_vec, _mse, scale = symmetric_quantize_vector(clean_vec, bits)
        scales.append(float(scale))
        set_unit_vector(modified_cache, unit, dequantized_vec)
        diff = dequantized_vec.float() - clean_vec.float()
        total_sq_error += float((diff * diff).sum().item())
        total_elements += int(diff.numel())
    mse = (total_sq_error / total_elements) if total_elements > 0 else 0.0
    scale_mean = (sum(scales) / len(scales)) if scales else 0.0
    scale_max = max(scales) if scales else 0.0
    return {
        "quantization_mse": float(mse),
        "num_quantized_elements": int(total_elements),
        "quantization_scale_mean": float(scale_mean),
        "quantization_scale_max": float(scale_max),
        "sample_shape": sample_shape,
        "sample_dtype": sample_dtype,
        "sample_device": sample_device,
    }


def build_selection_plan(strategy, budget_ratio, budget_mode, packed_cache, mixer_pairs, score_maps, seed, hybrid_alpha, rank=None, selection_cache=None, validate_mixed_quant_selector=False, mixed_quant_validate_k=128):
    strategy = STRATEGY_ALIASES.get(strategy, strategy)
    selection_cache = selection_cache if selection_cache is not None else {}
    inventory = selection_cache.get("inventory")
    if inventory is None:
        inventory = collect_cache_inventory(clone_cache_deep(packed_cache), mixer_pairs)
        selection_cache["inventory"] = inventory
    manageable_units = inventory["manageable_units"]
    total_cache_elements = int(inventory["total_cache_elements"])
    total_cache_bytes = int(inventory["total_cache_bytes"])
    manageable_state_elements = int(inventory["manageable_state_elements"])
    unmanaged_elements = int(inventory["unmanaged_elements"])
    unmanaged_bytes = int(inventory["unmanaged_bytes"])
    dtype_bytes = get_primary_state_dtype_bytes(packed_cache)
    if inventory["warnings"]:
        for warning in inventory["warnings"]:
            print(f"[WARN] {warning}")

    if strategy == "full_state":
        return {
            "inventory": inventory,
            "selected_units": list(manageable_units),
            "dropped_units": [],
            "kept_manageable_elements": manageable_state_elements,
            "total_cache_elements": total_cache_elements,
            "total_cache_bytes": total_cache_bytes,
            "manageable_state_elements": manageable_state_elements,
            "unmanaged_elements": unmanaged_elements,
            "unmanaged_bytes": unmanaged_bytes,
            "target_budget_ratio": 1.0,
            "priority_build_ms": 0.0,
            "mask_build_ms": 0.0,
        }

    if manageable_state_elements <= 0 or not manageable_units:
        raise RuntimeError("status=no_manageable_state: no manageable state units")

    fallback_reason = ""
    if strategy in MIXED_QUANTIZATION_STRATEGIES and budget_mode == "effective_storage":
        min_budget_ratio = minimum_supported_budget_ratio_for_quantization(strategy)
        if min_budget_ratio is not None and float(budget_ratio) < float(min_budget_ratio):
            raise RuntimeError(
                f"status=unsupported_budget: target budget_ratio={float(budget_ratio):.4f} is smaller than minimum supported ratio {float(min_budget_ratio):.4f} for {strategy}"
            )

    if strategy in RESIDUAL_CORRECTION_STRATEGIES:
        priority_build_ms = 0.0
        mask_build_ms = 0.0
        estimated_state_memory_bytes = estimate_storage_bytes_for_selection(
            strategy,
            inventory,
            [],
            list(manageable_units),
            dtype_bytes,
            rank=rank,
        )
        estimated_effective_budget_ratio = (estimated_state_memory_bytes / total_cache_bytes) if total_cache_bytes > 0 else 0.0
        return {
            "inventory": inventory,
            "selected_units": [],
            "dropped_units": list(manageable_units),
            "mixed_fp16_units": [],
            "mixed_int8_units": [],
            "mixed_int4_units": [],
            "kept_manageable_elements": 0,
            "total_cache_elements": total_cache_elements,
            "total_cache_bytes": total_cache_bytes,
            "manageable_state_elements": manageable_state_elements,
            "unmanaged_elements": unmanaged_elements,
            "unmanaged_bytes": unmanaged_bytes,
            "target_budget_ratio": float(budget_ratio),
            "budget_mode": budget_mode,
            "estimated_state_memory_bytes": int(estimated_state_memory_bytes),
            "estimated_effective_budget_ratio": float(estimated_effective_budget_ratio),
            "priority_build_ms": float(priority_build_ms),
            "mask_build_ms": float(mask_build_ms),
            "fallback_reason": fallback_reason,
        }

    priority_started = time.perf_counter()
    scoring_strategy = get_strategy_scoring_name(strategy)
    if strategy in (MIXED_QUANTIZATION_STRATEGIES | PRIORITY_MIXED_STRATEGIES) and not has_effective_head_sensitivity(score_maps):
        scoring_strategy = "state_norm_keep"
        fallback_reason = "missing_effective_head_sensitivity"
    priority_cache_key = (strategy, scoring_strategy, int(seed), float(hybrid_alpha if hybrid_alpha is not None else 1.0))
    cached_priority_entry = selection_cache.get("priority_entries", {}).get(priority_cache_key)
    if cached_priority_entry is None:
        state_norm_maps = selection_cache.get("state_norm_maps")
        if state_norm_maps is None:
            state_norm_maps = compute_state_norm_scores(packed_cache, mixer_pairs)
            selection_cache["state_norm_maps"] = state_norm_maps
        scored_units, _sensitivity_stats = build_priority_lists_for_units(
            manageable_units,
            scoring_strategy,
            score_maps,
            state_norm_maps,
            random.Random(seed),
            seed,
            hybrid_alpha=hybrid_alpha,
        )
        cached_priority_entry = {
            "scored_units": scored_units,
        }
        selection_cache.setdefault("priority_entries", {})[priority_cache_key] = cached_priority_entry
        priority_build_ms = (time.perf_counter() - priority_started) * 1000.0
    else:
        scored_units = cached_priority_entry["scored_units"]
        priority_build_ms = 0.0

    mask_started = time.perf_counter()
    if budget_mode == "retained_only":
        selected_units, kept_manageable_elements, _target_keep, _retained_budget_ratio, _budget_abs_error, _total_elements = select_units_exact_budget(
            scored_units,
            budget_ratio,
        )
        dropped_units = scored_units[len(selected_units):]
        estimated_state_memory_bytes = estimate_storage_bytes_for_selection(
            strategy,
            inventory,
            selected_units,
            dropped_units,
            dtype_bytes,
            rank=rank,
        )
        estimated_effective_budget_ratio = (estimated_state_memory_bytes / total_cache_bytes) if total_cache_bytes > 0 else 0.0
        mixed_fp16_units = selected_units if strategy in PRIORITY_MIXED_STRATEGIES else []
        mixed_int8_units = []
        mixed_int4_units = []
    elif budget_mode == "effective_storage":
        mixed_selector_cache = None
        if strategy in PRIORITY_MIXED_STRATEGIES:
            mixed_selector_cache = selection_cache.get("mixed_selector_entries", {}).get(priority_cache_key)
            if mixed_selector_cache is None:
                mixed_selector_cache = build_mixed_quant_selector_cache(scored_units)
                selection_cache.setdefault("mixed_selector_entries", {})[priority_cache_key] = mixed_selector_cache
        selection = select_units_for_effective_storage(
            scored_units,
            inventory,
            strategy,
            float(budget_ratio),
            dtype_bytes,
            rank=rank,
            mixed_selector_cache=mixed_selector_cache,
            validate_mixed_quant_selector=validate_mixed_quant_selector,
            mixed_quant_validate_k=mixed_quant_validate_k,
        )
        selected_units = list(selection.get("selected_units", []))
        kept_manageable_elements = int(selection.get("kept_manageable_elements", 0))
        estimated_effective_budget_ratio = float(selection.get("effective_budget_ratio", 0.0))
        estimated_state_memory_bytes = int(selection.get("estimated_state_memory_bytes", 0))
        dropped_units = list(selection.get("dropped_units", []))
        mixed_fp16_units = list(selection.get("mixed_fp16_units", []))
        mixed_int8_units = list(selection.get("mixed_int8_units", []))
        mixed_int4_units = list(selection.get("mixed_int4_units", []))
    else:
        raise ValueError(f"Unsupported budget_mode: {budget_mode}")

    selected_ids = {_unit_identity(unit) for unit in selected_units}
    if strategy not in PRIORITY_MIXED_STRATEGIES:
        dropped_units = [unit for unit in scored_units if _unit_identity(unit) not in selected_ids]
    mask_build_ms = (time.perf_counter() - mask_started) * 1000.0
    return {
        "inventory": inventory,
        "selected_units": selected_units,
        "dropped_units": dropped_units,
        "mixed_fp16_units": mixed_fp16_units,
        "mixed_int8_units": mixed_int8_units,
        "mixed_int4_units": mixed_int4_units,
        "kept_manageable_elements": int(kept_manageable_elements),
        "total_cache_elements": total_cache_elements,
        "total_cache_bytes": total_cache_bytes,
        "manageable_state_elements": manageable_state_elements,
        "unmanaged_elements": unmanaged_elements,
        "unmanaged_bytes": unmanaged_bytes,
        "target_budget_ratio": float(budget_ratio),
        "budget_mode": budget_mode,
        "estimated_state_memory_bytes": int(estimated_state_memory_bytes),
        "estimated_effective_budget_ratio": float(estimated_effective_budget_ratio),
        "priority_build_ms": float(priority_build_ms),
        "mask_build_ms": float(mask_build_ms),
        "fallback_reason": "; ".join([part for part in [fallback_reason, selection.get("validation_warning", "") if budget_mode == "effective_storage" else ""] if part]),
        "mixed_selector_method": selection.get("mixed_selector_method", "") if budget_mode == "effective_storage" else "",
        "mixed_selector_num_units": selection.get("mixed_selector_num_units", "") if budget_mode == "effective_storage" else "",
        "mixed_selector_elapsed_ms": selection.get("mixed_selector_elapsed_ms", "") if budget_mode == "effective_storage" else "",
        "mixed_selector_budget_abs_error": selection.get("mixed_selector_budget_abs_error", "") if budget_mode == "effective_storage" else "",
        "mixed_selector_fp16_units": selection.get("mixed_selector_fp16_units", "") if budget_mode == "effective_storage" else "",
        "mixed_selector_int8_units": selection.get("mixed_selector_int8_units", "") if budget_mode == "effective_storage" else "",
        "mixed_selector_int4_units": selection.get("mixed_selector_int4_units", "") if budget_mode == "effective_storage" else "",
        "mixed_selector_fp16_bytes": selection.get("mixed_selector_fp16_bytes", "") if budget_mode == "effective_storage" else "",
        "mixed_selector_int8_bytes": selection.get("mixed_selector_int8_bytes", "") if budget_mode == "effective_storage" else "",
        "mixed_selector_int4_bytes": selection.get("mixed_selector_int4_bytes", "") if budget_mode == "effective_storage" else "",
        "budget_under_min": selection.get("budget_under_min", "") if budget_mode == "effective_storage" else "",
        "budget_over_max": selection.get("budget_over_max", "") if budget_mode == "effective_storage" else "",
    }


def _stack_vectors(cache, units):
    if not units:
        return None
    return torch.stack([get_unit_vector(cache, unit).float() for unit in units], dim=0)


def reconstruct_by_mean(clean_cache, modified_cache, kept_units, dropped_units, use_scale, debug=False):
    kept_by_head = defaultdict(list)
    kept_by_layer = defaultdict(list)
    for unit in kept_units:
        kept_by_head[_group_key_head(unit)].append(unit)
        kept_by_layer[_group_key_layer(unit)].append(unit)

    global_kept = list(kept_units)
    stats_cache = {}
    stored_elements = 0
    reconstructed_elements = 0

    def resolve_group_units(drop_unit):
        head_key = _group_key_head(drop_unit)
        if kept_by_head.get(head_key):
            return ("head", head_key), kept_by_head[head_key]
        layer_key = _group_key_layer(drop_unit)
        if kept_by_layer.get(layer_key):
            return ("layer", layer_key), kept_by_layer[layer_key]
        return ("global", "all"), global_kept

    apply_started = time.perf_counter()
    for unit in dropped_units:
        stat_key, source_units = resolve_group_units(unit)
        if stat_key not in stats_cache:
            if not source_units:
                stats_cache[stat_key] = None
            else:
                matrix = _stack_vectors(clean_cache, source_units)
                mean_vec = matrix.mean(dim=0)
                if use_scale:
                    std_vec = matrix.std(dim=0, unbiased=False) if matrix.shape[0] > 1 else torch.zeros_like(mean_vec)
                    prototype = mean_vec + torch.sign(mean_vec - mean_vec.mean()) * std_vec
                    stats_cache[stat_key] = (prototype, mean_vec, std_vec)
                    stored_elements += int(mean_vec.numel() + std_vec.numel())
                else:
                    stats_cache[stat_key] = (mean_vec,)
                    stored_elements += int(mean_vec.numel())
        payload = stats_cache[stat_key]
        if payload is None:
            recon_vec = torch.zeros_like(get_unit_vector(clean_cache, unit).float())
        elif use_scale:
            recon_vec = payload[0]
        else:
            recon_vec = payload[0]
        set_unit_vector(modified_cache, unit, recon_vec)
        reconstructed_elements += int(unit["num_elements"])

    if debug:
        print(f"[reconstruct] method={'mean_scale' if use_scale else 'mean'} groups={len(stats_cache)} stored_elements={stored_elements} reconstructed_elements={reconstructed_elements}")
    reconstruct_apply_ms = (time.perf_counter() - apply_started) * 1000.0
    return stored_elements, reconstructed_elements, 0.0, reconstruct_apply_ms


def reconstruct_by_lowrank(clean_cache, modified_cache, dropped_units, rank, debug=False):
    dropped_by_layer = defaultdict(list)
    for unit in dropped_units:
        dropped_by_layer[_group_key_layer(unit)].append(unit)

    factor_elements = 0
    reconstructed_elements = 0
    lowrank_decompose_ms = 0.0
    reconstruct_apply_ms = 0.0
    for layer_idx, units in dropped_by_layer.items():
        if not units:
            continue
        matrix = _stack_vectors(clean_cache, units)
        if matrix is None or matrix.numel() == 0:
            continue
        rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
        rank_eff = min(int(rank), rows, cols)
        if rank_eff <= 0:
            continue
        decompose_started = time.perf_counter()
        try:
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        except Exception:
            u, s, vh = torch.linalg.svd(matrix.cpu(), full_matrices=False)
            u = u.to(matrix.device)
            s = s.to(matrix.device)
            vh = vh.to(matrix.device)
        lowrank_decompose_ms += (time.perf_counter() - decompose_started) * 1000.0
        apply_started = time.perf_counter()
        approx = (u[:, :rank_eff] * s[:rank_eff]) @ vh[:rank_eff, :]
        for unit, row in zip(units, approx, strict=False):
            set_unit_vector(modified_cache, unit, row)
            reconstructed_elements += int(unit["num_elements"])
        reconstruct_apply_ms += (time.perf_counter() - apply_started) * 1000.0
        factor_elements += int(rows * rank_eff + rank_eff * cols + rank_eff)
        if debug:
            print(f"[reconstruct] method=lowrank layer={layer_idx} rows={rows} cols={cols} rank_eff={rank_eff}")
    return factor_elements, reconstructed_elements, lowrank_decompose_ms, reconstruct_apply_ms


def apply_strategy_to_cache(clean_cache, plan, strategy, rank=None, debug=False):
    strategy = STRATEGY_ALIASES.get(strategy, strategy)
    modified_cache = clone_cache_deep(clean_cache)
    inventory = plan["inventory"]
    kept_units = list(plan["selected_units"])
    dropped_units = list(plan["dropped_units"])
    mixed_fp16_units = list(plan.get("mixed_fp16_units", []))
    mixed_int8_units = list(plan.get("mixed_int8_units", []))
    mixed_int4_units = list(plan.get("mixed_int4_units", []))
    quantization_bits = get_strategy_quantization_bits(strategy)
    quantization_mse = ""
    full_precision_fraction = ""
    quantized_fraction = ""
    quantization_scale_mean = ""
    quantization_scale_max = ""
    fallback_reason = plan.get("fallback_reason", "")
    lowrank_decompose_ms = 0.0
    reconstruct_apply_ms = 0.0
    quantize_ms = 0.0
    checkpoint_compress_ms = float(plan.get("priority_build_ms", 0.0) + plan.get("mask_build_ms", 0.0))
    checkpoint_restore_ms = 0.0
    reconstructed_state_elements = 0

    original_state_bytes = int(plan["total_cache_bytes"])
    unmanaged_bytes = int(plan["unmanaged_bytes"])
    manageable_fp_bytes = int(inventory.get("manageable_state_bytes", 0))
    dtype_bytes = get_primary_state_dtype_bytes(clean_cache)

    fp16_state_bytes = unmanaged_bytes
    int8_state_bytes = 0
    int4_state_bytes = 0
    scale_bytes = 0
    lowrank_factor_bytes = 0
    residual_bytes = 0

    exact_kept_elements = int(plan["unmanaged_elements"] + plan["kept_manageable_elements"])
    quantized_elements = 0
    reconstruction_units = []

    if strategy == "full_state":
        fp16_state_bytes = original_state_bytes
        full_precision_fraction = 1.0 if manageable_fp_bytes > 0 else ""
        quantized_fraction = 0.0 if manageable_fp_bytes > 0 else ""
        exact_kept_elements = int(plan["total_cache_elements"])
    elif strategy in {"random_zero_drop", "uniform_zero_drop", "state_norm_zero_drop", "hybrid_zero_drop"}:
        fp16_state_bytes += unit_num_bytes_sum(kept_units)
        full_precision_fraction = (unit_num_bytes_sum(kept_units) / manageable_fp_bytes) if manageable_fp_bytes > 0 else ""
        quantized_fraction = 0.0 if manageable_fp_bytes > 0 else ""
        reconstruction_units = dropped_units
        restore_started = time.perf_counter()
        zero_out_units(modified_cache, dropped_units)
        checkpoint_restore_ms += (time.perf_counter() - restore_started) * 1000.0
        reconstructed_state_elements = unit_num_elements_sum(dropped_units)
    elif strategy == "hybrid_zero_drop_plus_int8":
        compress_started = time.perf_counter()
        payload = build_quantized_unit_payload(clean_cache, kept_units, 8)
        quantize_ms += (time.perf_counter() - compress_started) * 1000.0
        checkpoint_compress_ms += quantize_ms
        int8_state_bytes += int(payload["data_bytes"])
        scale_bytes += int(payload["scale_bytes"])
        quantization_mse = payload["quantization_mse"]
        quantization_scale_mean = payload["quantization_scale_mean"]
        quantization_scale_max = payload["quantization_scale_max"]
        quantized_elements = int(payload["num_quantized_elements"])
        full_precision_fraction = 0.0 if manageable_fp_bytes > 0 else ""
        quantized_fraction = (quantized_elements / plan["manageable_state_elements"]) if plan["manageable_state_elements"] > 0 else ""
        reconstruction_units = list(inventory["manageable_units"])
        restore_started = time.perf_counter()
        zero_out_units(modified_cache, dropped_units)
        restored_quant_elements = restore_quantized_unit_payload(modified_cache, payload["entries"])
        checkpoint_restore_ms += (time.perf_counter() - restore_started) * 1000.0
        reconstructed_state_elements = int(unit_num_elements_sum(dropped_units) + restored_quant_elements)
    elif strategy in MIXED_QUANTIZATION_STRATEGIES and quantization_bits is not None:
        fp16_state_bytes += unit_num_bytes_sum(kept_units)
        compress_started = time.perf_counter()
        payload = build_quantized_unit_payload(clean_cache, dropped_units, quantization_bits)
        quantize_ms += (time.perf_counter() - compress_started) * 1000.0
        checkpoint_compress_ms += quantize_ms
        if quantization_bits == 8:
            int8_state_bytes += int(payload["data_bytes"])
        elif quantization_bits == 4:
            int4_state_bytes += int(payload["data_bytes"])
        scale_bytes += int(payload["scale_bytes"])
        quantization_mse = payload["quantization_mse"]
        quantization_scale_mean = payload["quantization_scale_mean"]
        quantization_scale_max = payload["quantization_scale_max"]
        quantized_elements = int(payload["num_quantized_elements"])
        full_precision_fraction = (unit_num_bytes_sum(kept_units) / manageable_fp_bytes) if manageable_fp_bytes > 0 else ""
        quantized_fraction = (quantized_elements / plan["manageable_state_elements"]) if plan["manageable_state_elements"] > 0 else ""
        reconstruction_units = dropped_units
        restore_started = time.perf_counter()
        reconstructed_state_elements = restore_quantized_unit_payload(modified_cache, payload["entries"])
        checkpoint_restore_ms += (time.perf_counter() - restore_started) * 1000.0
    elif strategy in PRIORITY_MIXED_STRATEGIES:
        fp16_state_bytes += unit_num_bytes_sum(mixed_fp16_units)
        exact_kept_elements = int(plan["unmanaged_elements"] + unit_num_elements_sum(mixed_fp16_units))
        compress_started = time.perf_counter()
        payload8 = build_quantized_unit_payload(clean_cache, mixed_int8_units, 8)
        payload4 = build_quantized_unit_payload(clean_cache, mixed_int4_units, 4)
        quantize_ms += (time.perf_counter() - compress_started) * 1000.0
        checkpoint_compress_ms += quantize_ms
        int8_state_bytes += int(payload8["data_bytes"])
        int4_state_bytes += int(payload4["data_bytes"])
        scale_bytes += int(payload8["scale_bytes"] + payload4["scale_bytes"])
        quantized_elements = int(payload8["num_quantized_elements"] + payload4["num_quantized_elements"])
        total_quant_sq_error = float(payload8["quantization_mse"] * payload8["num_quantized_elements"] + payload4["quantization_mse"] * payload4["num_quantized_elements"])
        quantization_mse = (total_quant_sq_error / quantized_elements) if quantized_elements > 0 else 0.0
        scales = [value for value in [payload8["quantization_scale_mean"], payload4["quantization_scale_mean"]] if value not in (None, "")]
        scale_maxs = [value for value in [payload8["quantization_scale_max"], payload4["quantization_scale_max"]] if value not in (None, "")]
        quantization_scale_mean = (sum(scales) / len(scales)) if scales else ""
        quantization_scale_max = max(scale_maxs) if scale_maxs else ""
        full_precision_fraction = (unit_num_bytes_sum(mixed_fp16_units) / manageable_fp_bytes) if manageable_fp_bytes > 0 else ""
        quantized_fraction = (quantized_elements / plan["manageable_state_elements"]) if plan["manageable_state_elements"] > 0 else ""
        reconstruction_units = list(mixed_int8_units) + list(mixed_int4_units)
        restore_started = time.perf_counter()
        reconstructed_state_elements = restore_quantized_unit_payload(modified_cache, payload8["entries"])
        reconstructed_state_elements += restore_quantized_unit_payload(modified_cache, payload4["entries"])
        checkpoint_restore_ms += (time.perf_counter() - restore_started) * 1000.0
    elif strategy == "mean_reconstruct":
        fp16_state_bytes += unit_num_bytes_sum(kept_units)
        compress_started = time.perf_counter()
        payload = build_mean_checkpoint_payload(clean_cache, kept_units, dropped_units, use_scale=False)
        checkpoint_compress_ms += (time.perf_counter() - compress_started) * 1000.0
        fp16_state_bytes += int(payload["stored_elements"] * dtype_bytes)
        full_precision_fraction = (unit_num_bytes_sum(kept_units) / manageable_fp_bytes) if manageable_fp_bytes > 0 else ""
        quantized_fraction = 0.0 if manageable_fp_bytes > 0 else ""
        reconstruction_units = dropped_units
        restore_started = time.perf_counter()
        reconstructed_state_elements = restore_mean_checkpoint_payload(modified_cache, payload)
        checkpoint_restore_ms += (time.perf_counter() - restore_started) * 1000.0
        reconstruct_apply_ms = checkpoint_restore_ms
    elif strategy == "mean_scale_reconstruct":
        fp16_state_bytes += unit_num_bytes_sum(kept_units)
        compress_started = time.perf_counter()
        payload = build_mean_checkpoint_payload(clean_cache, kept_units, dropped_units, use_scale=True)
        checkpoint_compress_ms += (time.perf_counter() - compress_started) * 1000.0
        fp16_state_bytes += int(payload["stored_elements"] * dtype_bytes)
        full_precision_fraction = (unit_num_bytes_sum(kept_units) / manageable_fp_bytes) if manageable_fp_bytes > 0 else ""
        quantized_fraction = 0.0 if manageable_fp_bytes > 0 else ""
        reconstruction_units = dropped_units
        restore_started = time.perf_counter()
        reconstructed_state_elements = restore_mean_checkpoint_payload(modified_cache, payload)
        checkpoint_restore_ms += (time.perf_counter() - restore_started) * 1000.0
        reconstruct_apply_ms = checkpoint_restore_ms
    elif strategy == "lowrank_reconstruct":
        fp16_state_bytes += unit_num_bytes_sum(kept_units)
        compress_started = time.perf_counter()
        payload = build_lowrank_checkpoint_payload(clean_cache, dropped_units, rank=int(rank or 0), factor_bits=None)
        checkpoint_compress_ms += (time.perf_counter() - compress_started) * 1000.0
        lowrank_decompose_ms = float(payload["lowrank_decompose_ms"])
        lowrank_factor_bytes = int(payload["lowrank_factor_bytes"])
        full_precision_fraction = (unit_num_bytes_sum(kept_units) / manageable_fp_bytes) if manageable_fp_bytes > 0 else ""
        quantized_fraction = 0.0 if manageable_fp_bytes > 0 else ""
        reconstruction_units = dropped_units
        restore_started = time.perf_counter()
        reconstructed_state_elements = restore_lowrank_checkpoint_payload(modified_cache, payload)
        checkpoint_restore_ms += (time.perf_counter() - restore_started) * 1000.0
        reconstruct_apply_ms = checkpoint_restore_ms
    elif strategy == "lowrank_reconstruct_plus_int8":
        fp16_state_bytes += unit_num_bytes_sum(kept_units)
        compress_started = time.perf_counter()
        payload = build_lowrank_checkpoint_payload(clean_cache, dropped_units, rank=int(rank or 0), factor_bits=8)
        compress_elapsed_ms = (time.perf_counter() - compress_started) * 1000.0
        checkpoint_compress_ms += compress_elapsed_ms
        lowrank_decompose_ms = float(payload["lowrank_decompose_ms"])
        lowrank_factor_bytes = int(payload["lowrank_factor_bytes"])
        scale_bytes += int(payload["scale_bytes"])
        quantization_mse = payload["quantization_mse"]
        quantization_scale_mean = payload["quantization_scale_mean"]
        quantization_scale_max = payload["quantization_scale_max"]
        full_precision_fraction = (unit_num_bytes_sum(kept_units) / manageable_fp_bytes) if manageable_fp_bytes > 0 else ""
        quantized_fraction = float(max(0.0, 1.0 - full_precision_fraction)) if full_precision_fraction not in (None, "") else ""
        reconstruction_units = dropped_units
        restore_started = time.perf_counter()
        reconstructed_state_elements = restore_lowrank_checkpoint_payload(modified_cache, payload)
        checkpoint_restore_ms += (time.perf_counter() - restore_started) * 1000.0
        reconstruct_apply_ms = checkpoint_restore_ms
    elif strategy == "residual_lowrank_plus_int8":
        exact_kept_elements = int(plan["unmanaged_elements"])
        compress_started = time.perf_counter()
        payload = build_residual_lowrank_plus_int8_payload(clean_cache, inventory["manageable_units"], rank=int(rank or 0))
        compress_elapsed_ms = (time.perf_counter() - compress_started) * 1000.0
        checkpoint_compress_ms += compress_elapsed_ms
        quantize_ms = compress_elapsed_ms
        lowrank_decompose_ms = float(payload["lowrank_decompose_ms"])
        int8_state_bytes += int(payload["base_quant_payload"]["data_bytes"])
        scale_bytes += int(payload["base_quant_payload"]["scale_bytes"])
        residual_bytes += int(payload["residual_bytes"])
        quantization_mse = payload["base_quant_payload"]["quantization_mse"]
        quantization_scale_mean = payload["base_quant_payload"]["quantization_scale_mean"]
        quantization_scale_max = payload["base_quant_payload"]["quantization_scale_max"]
        quantized_elements = int(payload["base_quant_payload"]["num_quantized_elements"])
        full_precision_fraction = 0.0 if manageable_fp_bytes > 0 else ""
        quantized_fraction = 1.0 if manageable_fp_bytes > 0 else ""
        reconstruction_units = list(inventory["manageable_units"])
        restore_started = time.perf_counter()
        reconstructed_state_elements = restore_residual_lowrank_plus_int8_payload(modified_cache, payload)
        checkpoint_restore_ms += (time.perf_counter() - restore_started) * 1000.0
        reconstruct_apply_ms = checkpoint_restore_ms
    elif strategy == "linear_head_reconstruct":
        raise NotImplementedError("status=not_implemented: linear_head_reconstruct not implemented yet")
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    transform_total_ms = float(checkpoint_compress_ms + checkpoint_restore_ms)
    storage = build_storage_breakdown(
        original_state_bytes=original_state_bytes,
        fp16_state_bytes=fp16_state_bytes,
        int8_state_bytes=int8_state_bytes,
        int4_state_bytes=int4_state_bytes,
        scale_bytes=scale_bytes,
        lowrank_factor_bytes=lowrank_factor_bytes,
        residual_bytes=residual_bytes,
    )
    reconstruction_mse = compute_reconstruction_mse(clean_cache, modified_cache, reconstruction_units)

    if debug and strategy in (MIXED_QUANTIZATION_STRATEGIES | PRIORITY_MIXED_STRATEGIES | {"hybrid_zero_drop_plus_int8", "residual_lowrank_plus_int8"}):
        print(
            "[checkpoint-debug] "
            f"strategy={strategy} compressed_state_bytes={storage['compressed_state_bytes']} "
            f"target_budget={float(plan.get('target_budget_ratio', 0.0)):.4f} effective_budget={float(storage['effective_budget_ratio']):.4f}"
        )
    return {
        "modified_cache": modified_cache,
        "checkpoint_compress_ms": float(checkpoint_compress_ms),
        "checkpoint_restore_ms": float(checkpoint_restore_ms),
        "transform_total_ms": float(transform_total_ms),
        "reconstruction_overhead_ms": float(transform_total_ms),
        "reconstruction_mse": reconstruction_mse,
        "quantization_bits": quantization_bits,
        "full_precision_fraction": full_precision_fraction,
        "quantized_fraction": quantized_fraction,
        "quantization_scale_mean": quantization_scale_mean,
        "quantization_scale_max": quantization_scale_max,
        "quantization_mse": quantization_mse,
        "fallback_reason": fallback_reason,
        "kept_state_elements": exact_kept_elements,
        "reconstructed_state_elements": int(reconstructed_state_elements),
        **storage,
        "priority_build_ms": float(plan.get("priority_build_ms", 0.0)),
        "mask_build_ms": float(plan.get("mask_build_ms", 0.0)),
        "lowrank_decompose_ms": float(lowrank_decompose_ms),
        "reconstruct_apply_ms": float(reconstruct_apply_ms),
        "quantize_ms": float(quantize_ms),
    }


def load_phase24_eval_rows(output_csv_path: Path):
    if not output_csv_path.exists():
        return []
    with output_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)

    eval_rows = []
    for row in raw_rows:
        strategy = STRATEGY_ALIASES.get(row.get("strategy", ""), row.get("strategy", ""))
        row["strategy"] = strategy
        if strategy == "__prompt__":
            continue
        loss_increase = local_safe_float(row.get("loss_increase"))
        if loss_increase is None:
            continue
        row["_loss_increase"] = loss_increase
        row["_continuation_ppl"] = local_safe_float(row.get("continuation_ppl"))
        row["_top1_agreement"] = local_safe_float(row.get("top1_agreement_to_full"))
        row["_reconstruction_mse"] = local_safe_float(row.get("reconstruction_mse"))
        row["_quantization_mse"] = local_safe_float(row.get("quantization_mse"))
        row["_compressed_state_bytes"] = local_safe_float(row.get("compressed_state_bytes"))
        row["_memory_reduction_ratio"] = local_safe_float(row.get("memory_reduction_ratio"))
        row["_checkpoint_compress_ms"] = local_safe_float(row.get("checkpoint_compress_ms"))
        row["_checkpoint_restore_ms"] = local_safe_float(row.get("checkpoint_restore_ms"))
        row["_transform_total_ms"] = local_safe_float(row.get("transform_total_ms"))
        row["_continuation_decode_ms"] = local_safe_float(row.get("continuation_decode_ms"))
        row["_session_capacity_gain"] = local_safe_float(row.get("session_capacity_gain"))
        row["_reconstruction_overhead_ms"] = local_safe_float(row.get("reconstruction_overhead_ms"))
        row["_budget_ratio"] = local_safe_float(row.get("budget_ratio"))
        row["_effective_budget_ratio"] = local_safe_float(row.get("effective_budget_ratio"))
        row["_rank"] = local_safe_int(row.get("rank"))
        row["_quantization_bits"] = local_safe_int(row.get("quantization_bits"))
        eval_rows.append(row)
    return eval_rows


def row_pair_compatible(row_a, row_b):
    if str(row_a.get("seed", "")) != str(row_b.get("seed", "")):
        return False
    if str(row_a.get("prompt_id", "")) != str(row_b.get("prompt_id", "")):
        return False
    if row_a.get("_budget_ratio") != row_b.get("_budget_ratio"):
        return False
    rank_b = row_b.get("_rank")
    if rank_b is not None and row_a.get("_rank") is not None and rank_b != row_a.get("_rank"):
        return False
    alpha_a = row_a.get("hybrid_alpha", "")
    alpha_b = row_b.get("hybrid_alpha", "")
    if alpha_b not in (None, "") and alpha_a not in (None, "") and str(alpha_a) != str(alpha_b):
        return False
    return True


def build_method_pair_stats(eval_rows, method_a, method_b):
    rows_a = [row for row in eval_rows if row.get("strategy") == method_a]
    rows_b = [row for row in eval_rows if row.get("strategy") == method_b]
    rows_b_by_key = defaultdict(list)
    for row in rows_b:
        key = (str(row.get("seed", "")), str(row.get("prompt_id", "")), row.get("_budget_ratio"))
        rows_b_by_key[key].append(row)

    grouped_pairs = defaultdict(list)
    for row_a in rows_a:
        key = (str(row_a.get("seed", "")), str(row_a.get("prompt_id", "")), row_a.get("_budget_ratio"))
        candidates = rows_b_by_key.get(key, [])
        matched = None
        for candidate in candidates:
            if row_pair_compatible(row_a, candidate):
                matched = candidate
                break
        if matched is None:
            continue
        grouped_pairs[(method_a, method_b, row_a.get("_rank"), row_a.get("_budget_ratio"))].append((row_a, matched))

    stats = []
    for (method_a, method_b, rank, budget_ratio), pairs in grouped_pairs.items():
        delta_loss = []
        delta_top1 = []
        delta_recon = []
        delta_mem = []
        for row_a, row_b in pairs:
            if row_a.get("_loss_increase") is not None and row_b.get("_loss_increase") is not None:
                delta_loss.append(row_a["_loss_increase"] - row_b["_loss_increase"])
            if row_a.get("_top1_agreement") is not None and row_b.get("_top1_agreement") is not None:
                delta_top1.append(row_a["_top1_agreement"] - row_b["_top1_agreement"])
            if row_a.get("_reconstruction_mse") is not None and row_b.get("_reconstruction_mse") is not None:
                delta_recon.append(row_a["_reconstruction_mse"] - row_b["_reconstruction_mse"])
            if row_a.get("_memory_reduction_ratio") is not None and row_b.get("_memory_reduction_ratio") is not None:
                delta_mem.append(row_a["_memory_reduction_ratio"] - row_b["_memory_reduction_ratio"])
        stats.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "rank": rank,
                "budget_ratio": budget_ratio,
                "num_pairs": len(pairs),
                "mean_delta_loss": (sum(delta_loss) / len(delta_loss)) if delta_loss else "",
                "win_rate_loss": (sum(1 for value in delta_loss if value < 0) / len(delta_loss)) if delta_loss else "",
                "mean_delta_top1": (sum(delta_top1) / len(delta_top1)) if delta_top1 else "",
                "win_rate_top1": (sum(1 for value in delta_top1 if value > 0) / len(delta_top1)) if delta_top1 else "",
                "mean_delta_reconstruction_mse": (sum(delta_recon) / len(delta_recon)) if delta_recon else "",
                "mean_delta_memory_reduction_ratio": (sum(delta_mem) / len(delta_mem)) if delta_mem else "",
            }
        )
    return stats


def dominates_summary_row(row_a, row_b):
    keys = {
        "loss": (local_safe_float(row_a.get("mean_loss_increase")), local_safe_float(row_b.get("mean_loss_increase")), -1),
        "top1": (local_safe_float(row_a.get("mean_top1_agreement")), local_safe_float(row_b.get("mean_top1_agreement")), 1),
        "memory": (local_safe_float(row_a.get("mean_memory_reduction_ratio")), local_safe_float(row_b.get("mean_memory_reduction_ratio")), 1),
        "restore": (local_safe_float(row_a.get("mean_checkpoint_restore_ms")), local_safe_float(row_b.get("mean_checkpoint_restore_ms")), -1),
    }
    if any(value_a is None or value_b is None for value_a, value_b, _ in keys.values()):
        return False
    no_worse = True
    strictly_better = False
    for value_a, value_b, direction in keys.values():
        if direction > 0:
            if value_a < value_b:
                no_worse = False
                break
            if value_a > value_b:
                strictly_better = True
        else:
            if value_a > value_b:
                no_worse = False
                break
            if value_a < value_b:
                strictly_better = True
    return no_worse and strictly_better


def assign_pareto_layers(summary_rows):
    by_budget = defaultdict(list)
    for row in summary_rows:
        by_budget[row.get("budget_ratio")].append(row)
    for rows in by_budget.values():
        remaining = list(rows)
        layer_idx = 1
        while remaining:
            frontier = []
            for candidate in remaining:
                if not any(other is not candidate and dominates_summary_row(other, candidate) for other in remaining):
                    frontier.append(candidate)
            for candidate in frontier:
                candidate["rank_by_pareto"] = layer_idx
            remaining = [candidate for candidate in remaining if candidate not in frontier]
            layer_idx += 1
        for row in rows:
            if row.get("rank_by_pareto") in (None, ""):
                row["rank_by_pareto"] = layer_idx


def build_phase24_summary(output_csv_path: Path, summary_output_path: Path, paired_output_path: Path, paper_table_output_path: Path, pareto_output_path: Path, run_id: str):
    if not output_csv_path.exists():
        print(f"[WARN] output csv missing, skip summary: {output_csv_path}")
        return []

    eval_rows = load_phase24_eval_rows(output_csv_path)
    if not eval_rows:
        print(f"[WARN] no valid eval rows in {output_csv_path}")
        write_summary(summary_output_path, [])
        write_named_rows(paired_output_path, [], PAIRED_FIELDNAMES)
        write_named_rows(paper_table_output_path, [], PAPER_TABLE_FIELDNAMES)
        write_named_rows(pareto_output_path, [], PARETO_FIELDNAMES)
        return []

    grouped = defaultdict(list)
    for row in eval_rows:
        key = (
            row.get("strategy", ""),
            row.get("rank", ""),
            row.get("hybrid_alpha", ""),
            row.get("budget_ratio", ""),
            row.get("quantization_bits", ""),
        )
        grouped[key].append(row)

    summary_rows = []
    summary_lookup = {}
    for key, rows in grouped.items():
        strategy, rank, hybrid_alpha, budget_ratio, quantization_bits = key
        loss_vals = [row["_loss_increase"] for row in rows if row.get("_loss_increase") is not None]
        ppl_vals = [row["_continuation_ppl"] for row in rows if row.get("_continuation_ppl") is not None]
        top1_vals = [row["_top1_agreement"] for row in rows if row.get("_top1_agreement") is not None]
        recon_vals = [row["_reconstruction_mse"] for row in rows if row.get("_reconstruction_mse") is not None]
        quant_vals = [row["_quantization_mse"] for row in rows if row.get("_quantization_mse") is not None]
        compressed_vals = [row["_compressed_state_bytes"] for row in rows if row.get("_compressed_state_bytes") is not None]
        mem_vals = [row["_memory_reduction_ratio"] for row in rows if row.get("_memory_reduction_ratio") is not None]
        checkpoint_compress_vals = [row["_checkpoint_compress_ms"] for row in rows if row.get("_checkpoint_compress_ms") is not None]
        checkpoint_restore_vals = [row["_checkpoint_restore_ms"] for row in rows if row.get("_checkpoint_restore_ms") is not None]
        transform_total_vals = [row["_transform_total_ms"] for row in rows if row.get("_transform_total_ms") is not None]
        continuation_decode_vals = [row["_continuation_decode_ms"] for row in rows if row.get("_continuation_decode_ms") is not None]
        session_gain_vals = [row["_session_capacity_gain"] for row in rows if row.get("_session_capacity_gain") is not None]
        eff_budget_vals = [row["_effective_budget_ratio"] for row in rows if row.get("_effective_budget_ratio") is not None]
        mean_loss, std_loss = local_mean_std(loss_vals)
        mean_ppl, _ = local_mean_std(ppl_vals)
        mean_top1, std_top1 = local_mean_std(top1_vals)
        mean_recon, _ = local_mean_std(recon_vals)
        mean_quant, _ = local_mean_std(quant_vals)
        mean_compressed, _ = local_mean_std(compressed_vals)
        mean_mem, _ = local_mean_std(mem_vals)
        mean_checkpoint_compress, _ = local_mean_std(checkpoint_compress_vals)
        mean_checkpoint_restore, _ = local_mean_std(checkpoint_restore_vals)
        mean_transform_total, _ = local_mean_std(transform_total_vals)
        mean_continuation_decode, _ = local_mean_std(continuation_decode_vals)
        mean_session_gain, _ = local_mean_std(session_gain_vals)
        mean_eff_budget, _ = local_mean_std(eff_budget_vals)
        warning_parts = []
        target_budget = local_safe_float(budget_ratio)
        if target_budget is not None and mean_eff_budget not in (None, "") and strategy != "full_state":
            if abs(float(mean_eff_budget) - float(target_budget)) > 0.03:
                warning_parts.append("effective_budget_not_close_to_target")
        summary = {
            "run_id": run_id,
            "strategy": strategy,
            "rank": rank,
            "hybrid_alpha": hybrid_alpha,
            "budget_ratio": target_budget,
            "quantization_bits": local_safe_int(quantization_bits),
            "summary": PHASE24_SUMMARY_TEXT,
            "mean_effective_budget_ratio": mean_eff_budget,
            "mean_loss_increase": mean_loss,
            "std_loss_increase": std_loss,
            "mean_continuation_ppl": mean_ppl,
            "mean_top1_agreement": mean_top1,
            "std_top1_agreement": std_top1,
            "mean_reconstruction_mse": mean_recon,
            "mean_quantization_mse": mean_quant,
            "mean_compressed_state_bytes": mean_compressed,
            "mean_memory_reduction_ratio": mean_mem,
            "mean_checkpoint_compress_ms": mean_checkpoint_compress,
            "mean_checkpoint_restore_ms": mean_checkpoint_restore,
            "mean_transform_total_ms": mean_transform_total,
            "mean_continuation_decode_ms": mean_continuation_decode,
            "mean_session_capacity_gain": mean_session_gain,
            "num_prompts": len({row.get("prompt_id", "") for row in rows}),
            "num_seeds": len({row.get("seed", "") for row in rows}),
            "win_rate_vs_state_quant_int8": "",
            "win_rate_vs_hybrid_zero_drop": "",
            "rank_by_loss": "",
            "rank_by_pareto": "",
            "warning": ";".join(warning_parts),
        }
        summary_rows.append(summary)
        summary_lookup[(strategy, rank, hybrid_alpha, target_budget, local_safe_int(quantization_bits))] = summary

    by_budget = defaultdict(list)
    for row in summary_rows:
        by_budget[row.get("budget_ratio")].append(row)
    for budget_ratio, rows in by_budget.items():
        rows.sort(key=lambda item: (float("inf") if item["mean_loss_increase"] in (None, "") else item["mean_loss_increase"], str(item["strategy"]), str(item["rank"])))
        for rank_idx, row in enumerate(rows, start=1):
            row["rank_by_loss"] = rank_idx

    assign_pareto_layers(summary_rows)

    paired_rows = []
    pair_specs = [
        ("lowrank_reconstruct", "hybrid_zero_drop"),
        ("lowrank_reconstruct", "random_zero_drop"),
        ("lowrank_reconstruct", "uniform_zero_drop"),
        ("lowrank_reconstruct", "state_norm_zero_drop"),
        ("lowrank_reconstruct", "state_quant_int8"),
        ("lowrank_reconstruct", "state_quant_int4"),
    ]
    for method_a, method_b in pair_specs:
        for stat in build_method_pair_stats(eval_rows, method_a, method_b):
            paired_rows.append({"run_id": run_id, **stat})
    paired_rows.sort(key=lambda row: (row.get("budget_ratio"), float("inf") if row.get("rank") is None else row.get("rank"), row.get("method_b")))
    write_named_rows(paired_output_path, paired_rows, PAIRED_FIELDNAMES)

    pair_lookup = {
        (row["method_a"], row["method_b"], row["rank"], row["budget_ratio"]): row
        for row in paired_rows
    }
    for summary in summary_rows:
        baseline_hybrid = pair_lookup.get((summary.get("strategy"), "hybrid_zero_drop", local_safe_int(summary.get("rank")), summary.get("budget_ratio")))
        baseline_quant = pair_lookup.get((summary.get("strategy"), "state_quant_int8", local_safe_int(summary.get("rank")), summary.get("budget_ratio")))
        if baseline_hybrid is None:
            method_pair_rows = build_method_pair_stats(eval_rows, summary.get("strategy"), "hybrid_zero_drop")
            baseline_hybrid = next((row for row in method_pair_rows if row.get("rank") == local_safe_int(summary.get("rank")) and row.get("budget_ratio") == summary.get("budget_ratio")), None)
        if baseline_quant is None:
            method_pair_rows = build_method_pair_stats(eval_rows, summary.get("strategy"), "state_quant_int8")
            baseline_quant = next((row for row in method_pair_rows if row.get("rank") == local_safe_int(summary.get("rank")) and row.get("budget_ratio") == summary.get("budget_ratio")), None)
        summary["win_rate_vs_hybrid_zero_drop"] = "" if summary.get("strategy") == "hybrid_zero_drop" else (baseline_hybrid.get("win_rate_loss") if baseline_hybrid else "")
        summary["win_rate_vs_state_quant_int8"] = "" if summary.get("strategy") == "state_quant_int8" else (baseline_quant.get("win_rate_loss") if baseline_quant else "")

    write_summary(summary_output_path, summary_rows)

    paper_rows = []
    for summary in sorted(summary_rows, key=lambda row: (row.get("budget_ratio"), row.get("rank_by_loss"), str(row.get("strategy")), str(row.get("rank")))):
        baseline_hybrid = pair_lookup.get((summary.get("strategy"), "hybrid_zero_drop", local_safe_int(summary.get("rank")), summary.get("budget_ratio")))
        baseline_quant = pair_lookup.get((summary.get("strategy"), "state_quant_int8", local_safe_int(summary.get("rank")), summary.get("budget_ratio")))
        if baseline_hybrid is None:
            method_pair_rows = build_method_pair_stats(eval_rows, summary.get("strategy"), "hybrid_zero_drop")
            baseline_hybrid = next((row for row in method_pair_rows if row.get("rank") == local_safe_int(summary.get("rank")) and row.get("budget_ratio") == summary.get("budget_ratio")), None)
        if baseline_quant is None:
            method_pair_rows = build_method_pair_stats(eval_rows, summary.get("strategy"), "state_quant_int8")
            baseline_quant = next((row for row in method_pair_rows if row.get("rank") == local_safe_int(summary.get("rank")) and row.get("budget_ratio") == summary.get("budget_ratio")), None)
        paper_rows.append(
            {
                "budget_ratio": summary.get("budget_ratio"),
                "method": summary.get("strategy"),
                "rank": summary.get("rank"),
                "quantization_bits": summary.get("quantization_bits"),
                "summary": summary.get("summary"),
                "mean_loss_increase": summary.get("mean_loss_increase"),
                "std_loss_increase": summary.get("std_loss_increase"),
                "mean_top1_agreement": summary.get("mean_top1_agreement"),
                "std_top1_agreement": summary.get("std_top1_agreement"),
                "mean_reconstruction_mse": summary.get("mean_reconstruction_mse"),
                "mean_quantization_mse": summary.get("mean_quantization_mse"),
                "mean_compressed_state_bytes": summary.get("mean_compressed_state_bytes"),
                "mean_memory_reduction_ratio": summary.get("mean_memory_reduction_ratio"),
                "mean_checkpoint_compress_ms": summary.get("mean_checkpoint_compress_ms"),
                "mean_checkpoint_restore_ms": summary.get("mean_checkpoint_restore_ms"),
                "mean_continuation_decode_ms": summary.get("mean_continuation_decode_ms"),
                "mean_session_capacity_gain": summary.get("mean_session_capacity_gain"),
                "win_rate_vs_hybrid_zero_drop": summary.get("win_rate_vs_hybrid_zero_drop"),
                "win_rate_vs_state_quant_int8": summary.get("win_rate_vs_state_quant_int8"),
                "rank_by_loss": summary.get("rank_by_loss"),
                "rank_by_pareto": summary.get("rank_by_pareto"),
            }
        )
    write_named_rows(paper_table_output_path, paper_rows, PAPER_TABLE_FIELDNAMES)

    pareto_rows = []
    for summary in sorted(summary_rows, key=lambda row: (row.get("budget_ratio"), row.get("rank_by_pareto"), str(row.get("strategy")), str(row.get("rank")))):
        pareto_rows.append(
            {
                "run_id": run_id,
                "budget_ratio": summary.get("budget_ratio"),
                "method": summary.get("strategy"),
                "rank": summary.get("rank"),
                "quantization_bits": summary.get("quantization_bits"),
                "summary": summary.get("summary"),
                "mean_effective_budget_ratio": summary.get("mean_effective_budget_ratio"),
                "mean_loss_increase": summary.get("mean_loss_increase"),
                "mean_top1_agreement": summary.get("mean_top1_agreement"),
                "mean_compressed_state_bytes": summary.get("mean_compressed_state_bytes"),
                "mean_memory_reduction_ratio": summary.get("mean_memory_reduction_ratio"),
                "mean_checkpoint_compress_ms": summary.get("mean_checkpoint_compress_ms"),
                "mean_checkpoint_restore_ms": summary.get("mean_checkpoint_restore_ms"),
                "is_pareto_frontier": bool(summary.get("rank_by_pareto") == 1),
                "pareto_layer": summary.get("rank_by_pareto"),
            }
        )
    write_named_rows(pareto_output_path, pareto_rows, PARETO_FIELDNAMES)
    print(f"Saved summary to {summary_output_path}")
    print(f"Saved paired comparison to {paired_output_path}")
    print(f"Saved paper table to {paper_table_output_path}")
    print(f"Saved pareto summary to {pareto_output_path}")
    return summary_rows


def run_single_invocation(args, run_id, output_path):
    require_runtime_deps()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    score_info = parse_phase22_tables(args.phase22_layer_csv, args.phase22_head_csv, args.phase22_group_csv)
    score_maps = build_score_maps(score_info)

    ensure_local_hf_cache(Path(args.output_dir))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_loader = build_model(device=device, dtype=torch.float16 if device == "cuda" else torch.float32, local_files_only=False)
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    print(f"Loading model: {MODEL_NAME}")
    print(f"Model loader: {model_loader}")
    configure_mamba2_for_equivalence(model, disable_mem_eff_path=args.disable_mem_eff_path)

    mixer_pairs = collect_mamba2_mixers(model)
    if not mixer_pairs:
        raise RuntimeError("No Mamba2 mixers found in model")
    print(f"Collected {len(mixer_pairs)} Mamba2 mixers")

    eval_samples = build_synthetic_or_wikitext_samples(args, tokenizer)
    for sample in eval_samples:
        validate_sample(sample, args)
    print(f"Prepared {len(eval_samples)} eval samples")

    rows_to_write = []
    strategy_variants = expand_strategy_variants(args.strategies, args.hybrid_alpha, args.rank)

    for sample in eval_samples:
        prompt_id = int(sample["prompt_id"])
        prefix_ids = sample["prefix_ids"]
        continuation_ids = sample["continuation_ids"]
        prompt_row = {
            "run_id": run_id,
            "seed": args.seed,
            "dataset": sample["dataset"],
            "strategy": "__prompt__",
            "reconstruction_method": "",
            "rank": "",
            "hybrid_alpha": "",
            "budget_ratio": 1.0,
            "budget_mode": args.budget_mode,
            "effective_budget_ratio": 1.0,
            "summary": PHASE24_SUMMARY_TEXT,
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
            "reconstruction_mse": "",
            "quantization_bits": "",
            "full_precision_fraction": "",
            "quantized_fraction": "",
            "quantization_scale_mean": "",
            "quantization_scale_max": "",
            "quantization_mse": "",
            "fallback_reason": "",
            "original_state_elements": "",
            "kept_state_elements": "",
            "reconstructed_state_elements": "",
            "original_state_bytes": "",
            "compressed_state_bytes": "",
            "estimated_state_memory_bytes": "",
            "fp16_state_bytes": "",
            "int8_state_bytes": "",
            "int4_state_bytes": "",
            "scale_bytes": "",
            "lowrank_factor_bytes": "",
            "residual_bytes": "",
            "compression_ratio": "",
            "memory_reduction_ratio": "",
            "max_sessions_full_state_under_1GB": "",
            "max_sessions_compressed_under_1GB": "",
            "session_capacity_gain": "",
            "checkpoint_compress_ms": "",
            "checkpoint_restore_ms": "",
            "transform_total_ms": "",
            "continuation_decode_ms": "",
            "reconstruction_overhead_ms": "",
            "priority_build_ms": "",
            "mask_build_ms": "",
            "lowrank_decompose_ms": "",
            "reconstruct_apply_ms": "",
            "quantize_ms": "",
            "decode_ms": "",
            "average_decode_latency_ms": "",
            "tokens_per_second": "",
            "clean_cache_changed": "",
            "alias_check_pass": "",
            "full_repeat_check": "",
            "status": "started",
            "error_msg": "",
        }
        rows_to_write.append(prompt_row)
        if args.append_each_prompt:
            write_rows(output_path, rows_to_write, append=True)
            rows_to_write = []

        try:
            reference_prefill_logits, reference_cache = prefill_prefix(
                model,
                prefix_ids,
                args.prefix_len + args.continuation_len,
                device,
                args.cache_device,
            )
            ensure_tensor_like_cache_shapes(reference_cache, mixer_pairs)
            clean_cache = clone_cache_deep(reference_cache)
            clean_fp_before = cache_fingerprint_signature(clean_cache)
            full_logits, full_latency_ms, _ = run_full_state_from_clean(
                model,
                clean_cache,
                reference_prefill_logits,
                continuation_ids,
                device,
            )
            full_loss, full_ppl = compute_loss_and_ppl(full_logits, continuation_ids)
            original_state_elements = 0
            for layer_idx, _mixer in mixer_pairs:
                if layer_idx not in clean_cache["key_value_memory_dict"]:
                    continue
                conv_state, ssm_state = clean_cache["key_value_memory_dict"][layer_idx]
                original_state_elements += int(conv_state.numel() + ssm_state.numel())
            original_state_bytes = estimate_original_state_bytes(clean_cache, mixer_pairs)
            selection_cache = {}

            full_repeat_check = True
            if args.validate_full_repeat:
                repeat_logits, _, _ = run_full_state_from_clean(
                    model,
                    clean_cache,
                    reference_prefill_logits,
                    continuation_ids,
                    device,
                )
                full_repeat_check = torch.equal(full_logits, repeat_logits)
                if not full_repeat_check:
                    raise RuntimeError("status=full_repeat_mismatch: full_state pre-check failed")

            full_row = make_result_row(
                args=args,
                run_id=run_id,
                sample=sample,
                strategy="full_state",
                budget_ratio=1.0,
                effective_budget_ratio=1.0,
                full_loss=full_loss,
                continuation_loss=full_loss,
                full_ppl=full_ppl,
                continuation_ppl=full_ppl,
                logits_mse=0.0,
                top1_agreement=1.0,
                reconstruction_mse=0.0,
                original_state_elements=original_state_elements,
                kept_state_elements=original_state_elements,
                reconstructed_state_elements=0,
                original_state_bytes=original_state_bytes,
                compressed_state_bytes=original_state_bytes,
                estimated_state_memory_bytes=original_state_bytes,
                fp16_state_bytes=original_state_bytes,
                int8_state_bytes=0,
                int4_state_bytes=0,
                scale_bytes=0,
                lowrank_factor_bytes=0,
                residual_bytes=0,
                compression_ratio=1.0,
                memory_reduction_ratio=0.0,
                max_sessions_full_state_under_1GB=int(ONE_GB_BYTES // original_state_bytes) if original_state_bytes > 0 else 0,
                max_sessions_compressed_under_1GB=int(ONE_GB_BYTES // original_state_bytes) if original_state_bytes > 0 else 0,
                session_capacity_gain=1.0,
                checkpoint_compress_ms=0.0,
                checkpoint_restore_ms=0.0,
                transform_total_ms=0.0,
                continuation_decode_ms=full_latency_ms,
                reconstruction_overhead_ms=0.0,
                quantization_bits="",
                full_precision_fraction=1.0,
                quantized_fraction=0.0,
                quantization_scale_mean=0.0,
                quantization_scale_max=0.0,
                quantization_mse="",
                fallback_reason="",
                priority_build_ms=0.0,
                mask_build_ms=0.0,
                lowrank_decompose_ms=0.0,
                reconstruct_apply_ms=0.0,
                quantize_ms=0.0,
                decode_ms=full_latency_ms,
                average_decode_latency_ms=full_latency_ms,
                tokens_per_second=(args.continuation_len / (full_latency_ms / 1000.0)) if full_latency_ms > 0 else 0.0,
                clean_cache_changed=False,
                alias_check_pass=True,
                full_repeat_check=full_repeat_check,
                status="functional_pass",
                error_msg="",
            )
            rows_to_write.append(full_row)

            for strategy, hybrid_alpha, rank in strategy_variants:
                if strategy == "full_state":
                    continue
                for budget_ratio in args.budget_ratios:
                    try:
                        if strategy == "linear_head_reconstruct":
                            rows_to_write.append(
                                make_error_row(
                                    args,
                                    run_id,
                                    prompt_id,
                                    strategy,
                                    budget_ratio,
                                    error_msg="linear_head_reconstruct is intentionally left optional in this phase",
                                    hybrid_alpha=(hybrid_alpha if hybrid_alpha is not None else ""),
                                    rank=(rank if rank is not None else ""),
                                    status="not_implemented",
                                )
                            )
                            continue

                        plan = build_selection_plan(
                            strategy=strategy,
                            budget_ratio=float(budget_ratio),
                            budget_mode=args.budget_mode,
                            packed_cache=clean_cache,
                            mixer_pairs=mixer_pairs,
                            score_maps=score_maps,
                            seed=int(args.seed * 100003 + prompt_id),
                            hybrid_alpha=hybrid_alpha,
                            rank=rank,
                            selection_cache=selection_cache,
                            validate_mixed_quant_selector=args.validate_mixed_quant_selector,
                            mixed_quant_validate_k=args.mixed_quant_validate_k,
                        )
                        alias_check_pass_1, _ = check_cache_alias(clean_cache, clone_cache_deep(clean_cache))
                        apply_result = apply_strategy_to_cache(
                            clean_cache=clean_cache,
                            plan=plan,
                            strategy=strategy,
                            rank=rank,
                            debug=args.debug_reconstruction,
                        )
                        modified_cache = apply_result["modified_cache"]
                        alias_check_pass_2, _ = check_cache_alias(clean_cache, modified_cache)
                        candidate_logits, latency_ms, _ = decode_continuation(
                            model,
                            modified_cache,
                            reference_prefill_logits,
                            continuation_ids,
                            device,
                        )
                        continuation_loss, continuation_ppl = compute_loss_and_ppl(candidate_logits, continuation_ids)
                        logits_mse, top1_agreement, _ = compute_logits_metrics(full_logits, candidate_logits)
                        clean_cache_changed = fingerprint_changed(clean_fp_before, cache_fingerprint_signature(clean_cache))

                        full_repeat_check_value = full_repeat_check
                        if args.validate_full_repeat:
                            repeat_logits_after, _, _ = run_full_state_from_clean(
                                model,
                                clean_cache,
                                reference_prefill_logits,
                                continuation_ids,
                                device,
                            )
                            full_repeat_check_value = torch.equal(full_logits, repeat_logits_after)

                        row = make_result_row(
                            args=args,
                            run_id=run_id,
                            sample=sample,
                            strategy=strategy,
                            budget_ratio=float(budget_ratio),
                            effective_budget_ratio=float(apply_result["effective_budget_ratio"]),
                            full_loss=full_loss,
                            continuation_loss=continuation_loss,
                            full_ppl=full_ppl,
                            continuation_ppl=continuation_ppl,
                            logits_mse=logits_mse,
                            top1_agreement=top1_agreement,
                            reconstruction_mse=float(apply_result["reconstruction_mse"]),
                            original_state_elements=original_state_elements,
                            kept_state_elements=int(apply_result["kept_state_elements"]),
                            reconstructed_state_elements=int(apply_result["reconstructed_state_elements"]),
                            original_state_bytes=int(apply_result["original_state_bytes"]),
                            compressed_state_bytes=int(apply_result["compressed_state_bytes"]),
                            estimated_state_memory_bytes=int(apply_result["estimated_state_memory_bytes"]),
                            fp16_state_bytes=int(apply_result["fp16_state_bytes"]),
                            int8_state_bytes=int(apply_result["int8_state_bytes"]),
                            int4_state_bytes=int(apply_result["int4_state_bytes"]),
                            scale_bytes=int(apply_result["scale_bytes"]),
                            lowrank_factor_bytes=int(apply_result["lowrank_factor_bytes"]),
                            residual_bytes=int(apply_result["residual_bytes"]),
                            compression_ratio=float(apply_result["compression_ratio"]),
                            memory_reduction_ratio=float(apply_result["memory_reduction_ratio"]),
                            max_sessions_full_state_under_1GB=int(apply_result["max_sessions_full_state_under_1GB"]),
                            max_sessions_compressed_under_1GB=int(apply_result["max_sessions_compressed_under_1GB"]),
                            session_capacity_gain=apply_result["session_capacity_gain"],
                            checkpoint_compress_ms=float(apply_result["checkpoint_compress_ms"]),
                            checkpoint_restore_ms=float(apply_result["checkpoint_restore_ms"]),
                            transform_total_ms=float(apply_result["transform_total_ms"]),
                            continuation_decode_ms=latency_ms,
                            reconstruction_overhead_ms=float(apply_result["reconstruction_overhead_ms"]),
                            quantization_bits=(apply_result["quantization_bits"] if apply_result["quantization_bits"] is not None else ""),
                            full_precision_fraction=apply_result["full_precision_fraction"],
                            quantized_fraction=apply_result["quantized_fraction"],
                            quantization_scale_mean=apply_result["quantization_scale_mean"],
                            quantization_scale_max=apply_result["quantization_scale_max"],
                            quantization_mse=apply_result["quantization_mse"],
                            fallback_reason=apply_result["fallback_reason"],
                            priority_build_ms=float(apply_result["priority_build_ms"]),
                            mask_build_ms=float(apply_result["mask_build_ms"]),
                            lowrank_decompose_ms=float(apply_result["lowrank_decompose_ms"]),
                            reconstruct_apply_ms=float(apply_result["reconstruct_apply_ms"]),
                            quantize_ms=float(apply_result["quantize_ms"]),
                            decode_ms=latency_ms,
                            average_decode_latency_ms=latency_ms,
                            tokens_per_second=(args.continuation_len / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0,
                            mixed_selector_method=plan.get("mixed_selector_method", ""),
                            mixed_selector_num_units=plan.get("mixed_selector_num_units", ""),
                            mixed_selector_elapsed_ms=plan.get("mixed_selector_elapsed_ms", ""),
                            mixed_selector_budget_abs_error=plan.get("mixed_selector_budget_abs_error", ""),
                            mixed_selector_fp16_units=plan.get("mixed_selector_fp16_units", ""),
                            mixed_selector_int8_units=plan.get("mixed_selector_int8_units", ""),
                            mixed_selector_int4_units=plan.get("mixed_selector_int4_units", ""),
                            mixed_selector_fp16_bytes=plan.get("mixed_selector_fp16_bytes", ""),
                            mixed_selector_int8_bytes=plan.get("mixed_selector_int8_bytes", ""),
                            mixed_selector_int4_bytes=plan.get("mixed_selector_int4_bytes", ""),
                            budget_under_min=plan.get("budget_under_min", ""),
                            budget_over_max=plan.get("budget_over_max", ""),
                            clean_cache_changed=clean_cache_changed,
                            alias_check_pass=bool(alias_check_pass_1 and alias_check_pass_2),
                            full_repeat_check=full_repeat_check_value,
                            status="ok",
                            error_msg="",
                            hybrid_alpha=(hybrid_alpha if hybrid_alpha is not None else ""),
                            rank=(rank if rank is not None else ""),
                        )
                        rows_to_write.append(row)
                    except Exception as exc:
                        rows_to_write.append(
                            make_error_row(
                                args,
                                run_id,
                                prompt_id,
                                strategy,
                                budget_ratio,
                                error_msg=format_exception_message(exc, max_trace_lines=20),
                                hybrid_alpha=(hybrid_alpha if hybrid_alpha is not None else ""),
                                rank=(rank if rank is not None else ""),
                                status=(str(exc).split("status=")[-1].split(":")[0] if "status=" in str(exc) else "error"),
                            )
                        )

            if args.validate_full_repeat:
                repeat_logits_after_all, _, _ = run_full_state_from_clean(
                    model,
                    clean_cache,
                    reference_prefill_logits,
                    continuation_ids,
                    device,
                )
                if not torch.equal(full_logits, repeat_logits_after_all):
                    raise RuntimeError("status=full_repeat_mismatch: full_state post-check failed")
        except Exception as exc:
            rows_to_write.append(
                make_error_row(
                    args,
                    run_id,
                    prompt_id,
                    "__prompt__",
                    1.0,
                    error_msg=format_exception_message(exc, max_trace_lines=20),
                )
            )

        if rows_to_write:
            write_rows(output_path, rows_to_write, append=True)
            rows_to_write = []


def main():
    args = parse_args()
    args.strategies = normalize_strategies(args.strategies)
    run_id, output_path, summary_path, paired_path, paper_table_path, pareto_path = build_output_paths(args)
    if args.dry_run_plan:
        score_info = dry_run_score_info(args)
        print_dry_run_plan(args, score_info)
        return

    print(f"run_id: {run_id}")
    print(f"output_csv path: {output_path}")
    print(PHASE24_SUMMARY_TEXT)
    clear_previous_outputs(output_path, summary_path, paired_path, paper_table_path, pareto_path)
    seed_list = resolve_seed_list(args)
    for index, seed in enumerate(seed_list):
        seed_args = copy.deepcopy(args)
        seed_args.seed = int(seed)
        seed_args.output_csv = str(output_path)
        print(f"\n=== Phase24 seed {seed_args.seed} ({index + 1}/{len(seed_list)}) ===")
        run_single_invocation(seed_args, run_id, output_path)

    build_phase24_summary(output_path, summary_path, paired_path, paper_table_path, pareto_path, run_id)


if __name__ == "__main__":
    main()
