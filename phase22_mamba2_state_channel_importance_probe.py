import argparse
import csv
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

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
    summarize_logits_diff,
)
from phase21_mamba2_state_cache_permutation_equivalence import build_inference_params, synchronize_if_cuda


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


GROUP_SIZE = 8
PLAN_NUM_LAYERS = 24
PLAN_NUM_HEADS = 24
PLAN_NUM_GROUPS = 8

LAYER_FIELDNAMES = [
    "layer_idx",
    "state_norm",
    "loss_increase",
    "ppl_increase",
    "top1_drop",
    "ablation_level",
    "valid_sensitivity_count",
    "positive_loss_count",
    "max_loss_increase",
    "mean_positive_loss_increase",
    "sensitivity_valid",
    "warning",
    "probe_layers",
    "probe_heads",
    "probe_channel_groups",
    "ablation_levels",
    "cache_device",
    "sync_each_token",
    "decode_call_index",
    "elapsed_sec",
]

HEAD_FIELDNAMES = [
    "layer_idx",
    "head_id",
    "state_norm",
    "loss_increase",
    "rank",
    "ablation_level",
    "valid_sensitivity_count",
    "positive_loss_count",
    "max_loss_increase",
    "mean_positive_loss_increase",
    "sensitivity_valid",
    "warning",
    "probe_layers",
    "probe_heads",
    "probe_channel_groups",
    "ablation_levels",
    "cache_device",
    "sync_each_token",
    "decode_call_index",
    "elapsed_sec",
]

GROUP_FIELDNAMES = [
    "layer_idx",
    "head_id",
    "channel_group_id",
    "channel_start",
    "channel_end",
    "state_norm",
    "loss_increase",
    "rank",
    "ablation_level",
    "valid_sensitivity_count",
    "positive_loss_count",
    "max_loss_increase",
    "mean_positive_loss_increase",
    "sensitivity_valid",
    "warning",
    "probe_layers",
    "probe_heads",
    "probe_channel_groups",
    "ablation_levels",
    "cache_device",
    "sync_each_token",
    "decode_call_index",
    "elapsed_sec",
]

SUMMARY_FIELDNAMES = [
    "scope",
    "enabled_ablation_levels",
    "top_k_share",
    "entropy",
    "gini",
    "rank_stability",
    "spearman_state_norm_vs_loss",
    "kendall_state_norm_vs_loss",
    "random_ablation_loss_increase",
    "uniform_ablation_loss_increase",
    "top_sensitive_loss_increase",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=64)
    parser.add_argument("--prefix_len", type=int, default=512)
    parser.add_argument("--continuation_len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable_mem_eff_path", action="store_true")
    parser.add_argument("--score_only", type=str, default="none", choices=["none", "state_norm"])
    parser.add_argument("--probe_layers", type=str, default="all")
    parser.add_argument("--probe_heads", type=str, default="all")
    parser.add_argument("--probe_channel_groups", type=str, default="all")
    parser.add_argument("--select_top_layers_by_state_norm", type=int, default=0)
    parser.add_argument("--select_top_heads_by_state_norm", type=int, default=0)
    parser.add_argument("--select_top_groups_by_state_norm", type=int, default=0)
    parser.add_argument("--ablation_levels", nargs="+", default=["layer", "head", "group"])
    parser.add_argument("--fast_smoke", action="store_true")
    parser.add_argument("--dry_run_plan", action="store_true")
    parser.add_argument("--sync_each_token", action="store_true")
    parser.add_argument("--cache_device", type=str, choices=["cpu", "cuda"], default=None)
    parser.add_argument("--flush_every", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def parse_index_spec(spec: str, upper_bound: int):
    if spec == "all":
        return list(range(upper_bound))
    items = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            items.append(int(token))
    items = sorted(set(items))
    invalid = [item for item in items if item < 0 or item >= upper_bound]
    if invalid:
        raise ValueError(f"Invalid indices in spec '{spec}': {invalid}")
    return items


def normalize_ablation_levels(levels):
    normalized = []
    for level in levels:
        if level not in {"layer", "head", "group"}:
            raise ValueError(f"Unsupported ablation level: {level}")
        if level not in normalized:
            normalized.append(level)
    return normalized


def apply_fast_smoke_overrides(args):
    if not args.fast_smoke:
        return args
    args.num_prompts = 1
    args.prefix_len = 128
    args.continuation_len = 8
    args.probe_layers = "0,23"
    args.probe_heads = "0,12,23"
    args.probe_channel_groups = "0,7"
    args.ablation_levels = ["layer", "head", "group"]
    args.cache_device = "cuda"
    args.sync_each_token = False
    return args


def estimate_decode_calls(num_prompts, continuation_len, num_layers, num_heads, num_groups, ablation_levels):
    calls_per_prompt = 1
    if "layer" in ablation_levels:
        calls_per_prompt += num_layers
    if "head" in ablation_levels:
        calls_per_prompt += num_layers * num_heads
    if "group" in ablation_levels:
        calls_per_prompt += num_layers * num_heads * num_groups
    total_calls = num_prompts * calls_per_prompt
    total_token_steps = total_calls * continuation_len
    return calls_per_prompt, total_calls, total_token_steps


def print_dry_run_plan(args, probe_layers, probe_heads, probe_channel_groups, ablation_levels):
    calls_per_prompt, total_calls, total_token_steps = estimate_decode_calls(
        args.num_prompts,
        args.continuation_len,
        len(probe_layers),
        len(probe_heads),
        len(probe_channel_groups),
        ablation_levels,
    )
    print("phase22 dry run plan")
    print(f"  selected layers               : {probe_layers}")
    print(f"  selected heads                : {probe_heads}")
    print(f"  selected channel groups       : {probe_channel_groups}")
    print(f"  enabled ablation levels       : {ablation_levels}")
    print(f"  num_prompts                   : {args.num_prompts}")
    print(f"  continuation_len              : {args.continuation_len}")
    print(f"  estimated_decode_calls_prompt : {calls_per_prompt}")
    print(f"  estimated_total_decode_calls  : {total_calls}")
    print(f"  estimated_total_token_steps   : {total_token_steps}")
    if total_calls > 10000:
        print("[WARN] This plan is large; consider using --probe_layers/--probe_heads/--probe_channel_groups or --ablation_levels.")


def append_rows(csv_path: Path, fieldnames, rows):
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def build_output_paths(output_dir: Path, enabled_levels):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    paths = {
        "summary": output_dir / f"phase22_mamba2_state_importance_summary__levels-{'-'.join(enabled_levels)}__{timestamp}.csv",
        "summary_latest": output_dir / "phase22_latest_state_importance_summary.csv",
    }
    if "layer" in enabled_levels:
        paths["layer"] = output_dir / f"phase22_mamba2_state_layer_sensitivity__level-layer__{timestamp}.csv"
        paths["layer_latest"] = output_dir / "phase22_latest_layer_sensitivity.csv"
    if "head" in enabled_levels:
        paths["head"] = output_dir / f"phase22_mamba2_state_head_sensitivity__level-head__{timestamp}.csv"
        paths["head_latest"] = output_dir / "phase22_latest_head_sensitivity.csv"
    if "group" in enabled_levels:
        paths["group"] = output_dir / f"phase22_mamba2_state_channel_group_sensitivity__level-group__{timestamp}.csv"
        paths["group_latest"] = output_dir / "phase22_latest_group_sensitivity.csv"
    return paths


def copy_latest(src: Path, dst: Path):
    if not src.exists():
        return
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def compute_sensitivity_metadata(rows, value_field="loss_increase"):
    values = []
    positive = []
    for row in rows:
        try:
            value = float(row.get(value_field, 0.0))
        except Exception:
            continue
        if not math.isfinite(value):
            continue
        values.append(value)
        if value > 0:
            positive.append(value)
    max_loss = max(values) if values else 0.0
    mean_positive = (sum(positive) / len(positive)) if positive else 0.0
    sensitivity_valid = len(positive) > 0 and max_loss > 0.0
    warning = "" if sensitivity_valid else "all_zero_loss_increase"
    return {
        "valid_sensitivity_count": len(values),
        "positive_loss_count": len(positive),
        "max_loss_increase": max_loss,
        "mean_positive_loss_increase": mean_positive,
        "sensitivity_valid": sensitivity_valid,
        "warning": warning,
    }


def sample_long_token_windows(tokenizer, num_prompts: int, total_len: int, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "`datasets` is required for phase22_mamba2_state_channel_importance_probe.py. "
            "Please install Hugging Face datasets in the active environment."
        ) from exc

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = []
    for item in dataset:
        text = item["text"].strip()
        if text:
            texts.append(text)
    if not texts:
        raise RuntimeError("No non-empty texts found in WikiText-2 validation split")

    merged_text = "\n\n".join(texts)
    encoded = tokenizer(merged_text, return_tensors="pt")
    all_input_ids = encoded["input_ids"][0]
    if all_input_ids.numel() < total_len:
        raise ValueError(
            f"Tokenized validation text is shorter than required total_len={total_len}: "
            f"got {all_input_ids.numel()} tokens"
        )

    max_start = all_input_ids.numel() - total_len
    if max_start == 0:
        return [all_input_ids[:total_len].clone() for _ in range(num_prompts)]

    generator = random.Random(seed)
    candidate_starts = list(range(max_start + 1))
    generator.shuffle(candidate_starts)

    stride = max(1, total_len // 4)
    selected_starts = []
    for start in candidate_starts:
        if all(abs(start - chosen) >= stride for chosen in selected_starts):
            selected_starts.append(start)
        if len(selected_starts) >= num_prompts:
            break

    if len(selected_starts) < num_prompts:
        for start in candidate_starts:
            if start not in selected_starts:
                selected_starts.append(start)
            if len(selected_starts) >= num_prompts:
                break

    if len(selected_starts) < num_prompts:
        raise ValueError(f"Only collected {len(selected_starts)} windows, fewer than requested {num_prompts}")

    return [all_input_ids[start : start + total_len].clone() for start in selected_starts[:num_prompts]]


def clone_cache_entry(entry):
    if torch.is_tensor(entry):
        return entry.detach().clone()
    if isinstance(entry, tuple):
        return tuple(clone_cache_entry(item) for item in entry)
    if isinstance(entry, list):
        return [clone_cache_entry(item) for item in entry]
    if isinstance(entry, dict):
        return {key: clone_cache_entry(value) for key, value in entry.items()}
    return entry


def pack_inference_params_to_cpu(inference_params):
    packed = {
        "max_seqlen": inference_params.max_seqlen,
        "max_batch_size": inference_params.max_batch_size,
        "seqlen_offset": inference_params.seqlen_offset,
        "batch_size_offset": inference_params.batch_size_offset,
        "lengths_per_sample": None if inference_params.lengths_per_sample is None else inference_params.lengths_per_sample.detach().cpu().clone(),
        "key_value_memory_dict": {},
    }
    for layer_idx, layer_cache in inference_params.key_value_memory_dict.items():
        if isinstance(layer_cache, tuple):
            packed["key_value_memory_dict"][layer_idx] = tuple(item.detach().cpu().clone() for item in layer_cache)
        elif isinstance(layer_cache, dict):
            packed["key_value_memory_dict"][layer_idx] = {
                key: value.detach().cpu().clone() if torch.is_tensor(value) else clone_cache_entry(value)
                for key, value in layer_cache.items()
            }
        else:
            packed["key_value_memory_dict"][layer_idx] = clone_cache_entry(layer_cache)
    return packed


def restore_inference_params_from_cpu(packed, device: str):
    from mamba_ssm.utils.generation import InferenceParams

    restored = InferenceParams(
        max_seqlen=packed["max_seqlen"],
        max_batch_size=packed["max_batch_size"],
        seqlen_offset=packed["seqlen_offset"],
        batch_size_offset=packed["batch_size_offset"],
        key_value_memory_dict={},
        lengths_per_sample=None if packed["lengths_per_sample"] is None else packed["lengths_per_sample"].to(device),
    )
    for layer_idx, layer_cache in packed["key_value_memory_dict"].items():
        if isinstance(layer_cache, tuple):
            restored.key_value_memory_dict[layer_idx] = tuple(item.to(device) for item in layer_cache)
        elif isinstance(layer_cache, dict):
            restored.key_value_memory_dict[layer_idx] = {
                key: value.to(device) if torch.is_tensor(value) else clone_cache_entry(value)
                for key, value in layer_cache.items()
            }
        else:
            restored.key_value_memory_dict[layer_idx] = layer_cache.to(device) if torch.is_tensor(layer_cache) else clone_cache_entry(layer_cache)
    return restored


def pack_inference_params(inference_params, cache_device: str):
    if cache_device == "cuda":
        return {
            "kind": "gpu_live",
            "max_seqlen": inference_params.max_seqlen,
            "max_batch_size": inference_params.max_batch_size,
            "seqlen_offset": inference_params.seqlen_offset,
            "batch_size_offset": inference_params.batch_size_offset,
            "lengths_per_sample": None
            if inference_params.lengths_per_sample is None
            else inference_params.lengths_per_sample.detach().clone(),
            "key_value_memory_dict": clone_cache_entry(inference_params.key_value_memory_dict),
        }
    packed = pack_inference_params_to_cpu(inference_params)
    packed["kind"] = "cpu_packed"
    return packed


def restore_inference_params(packed, device: str):
    if packed.get("kind") == "gpu_live":
        from mamba_ssm.utils.generation import InferenceParams

        return InferenceParams(
            max_seqlen=packed["max_seqlen"],
            max_batch_size=packed["max_batch_size"],
            seqlen_offset=packed["seqlen_offset"],
            batch_size_offset=packed["batch_size_offset"],
            key_value_memory_dict=clone_cache_entry(packed["key_value_memory_dict"]),
            lengths_per_sample=None
            if packed["lengths_per_sample"] is None
            else packed["lengths_per_sample"].to(device),
        )
    return restore_inference_params_from_cpu(packed, device)


@torch.no_grad()
def prefill_context(model, prefix_ids: torch.Tensor, continuation_len: int, device: str, cache_device: str):
    context_ids = prefix_ids[:-1]
    inference_params = build_inference_params(max_seqlen=prefix_ids.numel() + continuation_len, batch_size=1)
    if context_ids.numel() > 0:
        batch = context_ids.unsqueeze(0).to(device)
        synchronize_if_cuda(device)
        _ = model(batch, inference_params=inference_params)
        synchronize_if_cuda(device)
    inference_params.seqlen_offset = context_ids.numel()
    return pack_inference_params(inference_params, cache_device=cache_device)


@torch.no_grad()
def decode_from_packed_cache(
    model,
    packed_cache,
    decode_inputs: torch.Tensor,
    continuation_targets: torch.Tensor,
    device: str,
    sync_each_token: bool = False,
):
    inference_params = restore_inference_params(packed_cache, device)
    logits_chunks = []
    synchronize_if_cuda(device)
    for token in decode_inputs:
        token_batch = token.view(1, 1).to(device)
        if sync_each_token:
            synchronize_if_cuda(device)
        outputs = model(token_batch, inference_params=inference_params, num_last_tokens=1)
        if sync_each_token:
            synchronize_if_cuda(device)
        logits_chunks.append(outputs.logits.float().cpu())
        inference_params.seqlen_offset += 1
    synchronize_if_cuda(device)
    logits = torch.cat(logits_chunks, dim=1)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), continuation_targets.unsqueeze(0).reshape(-1), reduction="mean").item()
    ppl = math.exp(loss)
    return logits, loss, ppl


def _combined_norm(parts):
    total = 0.0
    for tensor in parts:
        if tensor is None:
            continue
        total += float(torch.sum(tensor.float() ** 2).item())
    return math.sqrt(total)


def extract_state_norms(packed_cache, mixer_pairs):
    layer_norms = {}
    head_norms = {}
    group_norms = {}
    for layer_idx, mixer in mixer_pairs:
        if layer_idx not in packed_cache["key_value_memory_dict"]:
            continue
        conv_state, ssm_state = packed_cache["key_value_memory_dict"][layer_idx]
        d_inner = int(mixer.d_inner)
        nheads = int(mixer.nheads)
        headdim = int(mixer.headdim)
        conv_x = conv_state[:, :d_inner, :]
        layer_norms[layer_idx] = _combined_norm([conv_x, ssm_state])
        for head_id in range(nheads):
            c0 = head_id * headdim
            c1 = (head_id + 1) * headdim
            conv_head = conv_x[:, c0:c1, :]
            ssm_head = ssm_state[:, head_id, :, :]
            head_norms[(layer_idx, head_id)] = _combined_norm([conv_head, ssm_head])
            for group_id, start in enumerate(range(0, headdim, GROUP_SIZE)):
                end = min(start + GROUP_SIZE, headdim)
                conv_group = conv_x[:, c0 + start : c0 + end, :]
                ssm_group = ssm_state[:, head_id, start:end, :]
                group_norms[(layer_idx, head_id, group_id)] = _combined_norm([conv_group, ssm_group])
    return layer_norms, head_norms, group_norms


def zero_layer_state_(packed_cache, mixer):
    layer_idx = int(mixer.layer_idx)
    conv_state, ssm_state = packed_cache["key_value_memory_dict"][layer_idx]
    conv_state[:, : int(mixer.d_inner), :] = 0
    ssm_state.zero_()


def zero_head_state_(packed_cache, mixer, head_id: int):
    layer_idx = int(mixer.layer_idx)
    headdim = int(mixer.headdim)
    c0 = head_id * headdim
    c1 = (head_id + 1) * headdim
    conv_state, ssm_state = packed_cache["key_value_memory_dict"][layer_idx]
    conv_state[:, c0:c1, :] = 0
    ssm_state[:, head_id, :, :] = 0


def zero_group_state_(packed_cache, mixer, head_id: int, group_id: int):
    layer_idx = int(mixer.layer_idx)
    headdim = int(mixer.headdim)
    c0 = head_id * headdim
    start = group_id * GROUP_SIZE
    end = min(start + GROUP_SIZE, headdim)
    conv_state, ssm_state = packed_cache["key_value_memory_dict"][layer_idx]
    conv_state[:, c0 + start : c0 + end, :] = 0
    ssm_state[:, head_id, start:end, :] = 0


def rank_descending(rows, key_name: str):
    sorted_rows = sorted(rows, key=lambda row: row[key_name], reverse=True)
    for rank, row in enumerate(sorted_rows, start=1):
        row["rank"] = rank
    return sorted_rows


def _rankdata(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            ranks[order[pos]] = avg_rank
        start = end
    return ranks


def pearson_corr(xs, ys):
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def spearman_corr(xs, ys):
    return pearson_corr(_rankdata(xs), _rankdata(ys))


def kendall_tau(xs, ys):
    concordant = 0
    discordant = 0
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def concentration_stats(values, top_k: int):
    tensor = torch.tensor(values, dtype=torch.float32)
    total = tensor.sum().item()
    if total <= 0:
        return 0.0, 0.0, 0.0
    probs = tensor / total
    top_k_share = torch.topk(probs, k=min(top_k, probs.numel())).values.sum().item()
    nonzero = probs[probs > 0]
    entropy = float((-(nonzero * nonzero.log()).sum()).item())
    sorted_probs, _ = torch.sort(probs)
    n = sorted_probs.numel()
    weighted_sum = torch.sum(torch.arange(1, n + 1, dtype=torch.float32) * sorted_probs).item()
    gini = (2.0 * weighted_sum) / (n * sorted_probs.sum().item()) - (n + 1) / n
    return top_k_share, entropy, gini


def average_pairwise_spearman(rank_lists):
    if len(rank_lists) < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(len(rank_lists)):
        for j in range(i + 1, len(rank_lists)):
            total += spearman_corr(rank_lists[i], rank_lists[j])
            count += 1
    return total / count if count > 0 else 1.0


def average_by_key(rows, key_name: str, value_name: str):
    totals = defaultdict(lambda: [0.0, 0])
    for row in rows:
        key = row[key_name]
        totals[key][0] += float(row[value_name])
        totals[key][1] += 1
    return {key: total / count for key, (total, count) in totals.items() if count > 0}


def select_top_keys_by_average(rows, key_name: str, value_name: str, top_k: int, candidates=None):
    averages = average_by_key(rows, key_name, value_name)
    if candidates is not None:
        averages = {key: value for key, value in averages.items() if key in set(candidates)}
    ranked = sorted(averages.items(), key=lambda item: item[1], reverse=True)
    return [key for key, _ in ranked[:top_k]]


def mean_of_top_by_key(rows, key_name: str, value_name: str, top_k: int):
    if not rows:
        return 0.0
    ranked = sorted(rows, key=lambda row: row[key_name], reverse=True)
    selected = ranked[: max(1, min(top_k, len(ranked)))]
    return sum(float(row[value_name]) for row in selected) / len(selected)


def random_mean_of_k(rows, value_name: str, k: int, seed: int):
    if not rows:
        return 0.0
    k = max(1, min(k, len(rows)))
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    selected = shuffled[:k]
    return sum(float(row[value_name]) for row in selected) / len(selected)


def round_robin_layer_mean(rows, value_name: str, k: int):
    if not rows:
        return 0.0
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["layer_idx"]].append(row)
    for layer_rows in grouped.values():
        layer_rows.sort(key=lambda row: row[value_name], reverse=True)
    selected = []
    layer_cycle = sorted(grouped)
    positions = {layer_idx: 0 for layer_idx in layer_cycle}
    k = max(1, min(k, len(rows)))
    while len(selected) < k:
        progressed = False
        for layer_idx in layer_cycle:
            if len(selected) >= k:
                break
            pos = positions[layer_idx]
            if pos < len(grouped[layer_idx]):
                selected.append(grouped[layer_idx][pos])
                positions[layer_idx] += 1
                progressed = True
        if not progressed:
            break
    return sum(float(row[value_name]) for row in selected) / len(selected)


def main():
    args = parse_args()
    args = apply_fast_smoke_overrides(args)
    args.ablation_levels = normalize_ablation_levels(args.ablation_levels)
    if args.cache_device is None:
        args.cache_device = "cpu"

    probe_layers = parse_index_spec(args.probe_layers, PLAN_NUM_LAYERS)
    probe_heads = parse_index_spec(args.probe_heads, PLAN_NUM_HEADS)
    probe_channel_groups = parse_index_spec(args.probe_channel_groups, PLAN_NUM_GROUPS)
    print_dry_run_plan(args, probe_layers, probe_heads, probe_channel_groups, args.ablation_levels)
    if args.dry_run_plan:
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if args.cache_device == "cuda" and device != "cuda":
        print("[WARN] cache_device=cuda requested but CUDA is unavailable; falling back to cache_device=cpu.")
        args.cache_device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_local_hf_cache(output_dir)

    output_paths = build_output_paths(output_dir, args.ablation_levels)
    summary_csv = output_paths["summary"]

    print("grad_x_state: skipped (not implemented; cache gradients through inference path are unstable).")
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    model, loader_name = build_model(device=device, dtype=dtype)
    print(f"Model loader: {loader_name}")
    model = configure_mamba2_for_equivalence(model, disable_mem_eff_path=args.disable_mem_eff_path)
    mixer_pairs = collect_mamba2_mixers(model)
    if not mixer_pairs:
        raise RuntimeError("No Mamba2 mixer layers found")
    selected_mixer_pairs = [(layer_idx, mixer) for layer_idx, mixer in mixer_pairs if layer_idx in probe_layers]
    if not selected_mixer_pairs:
        raise RuntimeError("No mixers remain after applying --probe_layers")

    cache_path_label = "gpu_clone" if args.cache_device == "cuda" else "cpu_packed_restore"
    print(f"cache_path={cache_path_label}")

    calls_per_prompt, total_calls, total_token_steps = estimate_decode_calls(
        args.num_prompts,
        args.continuation_len,
        len(probe_layers),
        len(probe_heads),
        len(probe_channel_groups),
        args.ablation_levels,
    )
    print(f"Planned decode calls: {total_calls} total, {calls_per_prompt} per prompt")
    print(f"Planned token steps : {total_token_steps}")

    total_len = args.prefix_len + args.continuation_len
    print("Loading and tokenizing WikiText-2 validation samples")
    samples = sample_long_token_windows(tokenizer, num_prompts=args.num_prompts, total_len=total_len, seed=args.seed)
    prompt_records = []
    prompt_head_norm_ranks = []
    prompt_group_norm_ranks = []
    decode_call_index = 0
    run_start = time.perf_counter()
    score_only_state_norm = args.score_only == "state_norm"

    for prompt_id, sample in enumerate(samples):
        prefix_ids = sample[: args.prefix_len]
        continuation_ids = sample[args.prefix_len : total_len]
        decode_inputs = torch.cat([prefix_ids[-1:], continuation_ids[:-1]], dim=0)
        packed_cache = prefill_context(model, prefix_ids, args.continuation_len, device, cache_device=args.cache_device)
        baseline_logits, baseline_loss, baseline_ppl = decode_from_packed_cache(
            model,
            packed_cache,
            decode_inputs,
            continuation_ids,
            device,
            sync_each_token=args.sync_each_token,
        )
        decode_call_index += 1
        layer_norms, head_norms, group_norms = extract_state_norms(packed_cache, mixer_pairs)
        prompt_records.append(
            {
                "prompt_id": prompt_id,
                "packed_cache": packed_cache,
                "decode_inputs": decode_inputs.cpu(),
                "continuation_ids": continuation_ids.cpu(),
                "baseline_logits": baseline_logits,
                "baseline_loss": baseline_loss,
                "baseline_ppl": baseline_ppl,
                "layer_norms": layer_norms,
                "head_norms": head_norms,
                "group_norms": group_norms,
            }
        )
        prompt_head_norm_ranks.append([head_norms[key] for key in sorted(head_norms) if key[0] in probe_layers and key[1] in probe_heads])
        prompt_group_norm_ranks.append(
            [
                group_norms[key]
                for key in sorted(group_norms)
                if key[0] in probe_layers and key[1] in probe_heads and key[2] in probe_channel_groups
            ]
        )
        print(
            f"Prepared prompt {prompt_id + 1}/{len(samples)} baseline_loss={baseline_loss:.6f} "
            f"decode_calls={decode_call_index}/{total_calls}"
        )

    if not prompt_records:
        raise RuntimeError("No usable prompts collected for prefix+continuation evaluation")

    layer_agg = {layer_idx: {"state_norm": 0.0, "loss_increase": 0.0, "ppl_increase": 0.0, "top1_drop": 0.0} for layer_idx in probe_layers}
    head_agg = {}
    group_agg = {}
    for layer_idx, mixer in selected_mixer_pairs:
        for head_id in probe_heads:
            head_agg[(layer_idx, head_id)] = {"state_norm": 0.0, "loss_increase": 0.0}
            for group_id in probe_channel_groups:
                group_agg[(layer_idx, head_id, group_id)] = {"state_norm": 0.0, "loss_increase": 0.0}

    for record in prompt_records:
        for layer_idx, value in record["layer_norms"].items():
            if layer_idx in layer_agg:
                layer_agg[layer_idx]["state_norm"] += value
        for key, value in record["head_norms"].items():
            if key in head_agg:
                head_agg[key]["state_norm"] += value
        for key, value in record["group_norms"].items():
            if key in group_agg:
                group_agg[key]["state_norm"] += value

    num_prompts = len(prompt_records)
    for values in layer_agg.values():
        values["state_norm"] /= num_prompts
    for values in head_agg.values():
        values["state_norm"] /= num_prompts
    for values in group_agg.values():
        values["state_norm"] /= num_prompts

    if args.select_top_layers_by_state_norm > 0:
        ranked_layers = sorted(layer_agg.items(), key=lambda item: item[1]["state_norm"], reverse=True)
        probe_layers = [layer_idx for layer_idx, _ in ranked_layers[: min(args.select_top_layers_by_state_norm, len(ranked_layers))]]
        print(f"Selected top layers by state_norm: {probe_layers}")
    if args.select_top_heads_by_state_norm > 0:
        head_totals = defaultdict(float)
        for (layer_idx, head_id), stats in head_agg.items():
            head_totals[head_id] += stats["state_norm"]
        ranked_heads = sorted(head_totals.items(), key=lambda item: item[1], reverse=True)
        probe_heads = [head_id for head_id, _ in ranked_heads[: min(args.select_top_heads_by_state_norm, len(ranked_heads))]]
        print(f"Selected top heads by state_norm: {probe_heads}")
    if args.select_top_groups_by_state_norm > 0:
        group_totals = defaultdict(float)
        for (layer_idx, head_id, group_id), stats in group_agg.items():
            group_totals[group_id] += stats["state_norm"]
        ranked_groups = sorted(group_totals.items(), key=lambda item: item[1], reverse=True)
        probe_channel_groups = [group_id for group_id, _ in ranked_groups[: min(args.select_top_groups_by_state_norm, len(ranked_groups))]]
        print(f"Selected top channel groups by state_norm: {probe_channel_groups}")

    layer_agg = {layer_idx: layer_agg[layer_idx] for layer_idx in probe_layers if layer_idx in layer_agg}
    head_agg = {
        key: value
        for key, value in head_agg.items()
        if key[0] in probe_layers and key[1] in probe_heads
    }
    group_agg = {
        key: value
        for key, value in group_agg.items()
        if key[0] in probe_layers and key[1] in probe_heads and key[2] in probe_channel_groups
    }
    selected_mixer_pairs = [(layer_idx, mixer) for layer_idx, mixer in mixer_pairs if layer_idx in probe_layers]

    def make_row_meta(decode_idx):
        return {
            "probe_layers": ",".join(map(str, probe_layers)),
            "probe_heads": ",".join(map(str, probe_heads)),
            "probe_channel_groups": ",".join(map(str, probe_channel_groups)),
            "ablation_levels": ",".join(args.ablation_levels),
            "cache_device": args.cache_device,
            "sync_each_token": str(bool(args.sync_each_token)),
            "decode_call_index": decode_idx,
            "elapsed_sec": f"{time.perf_counter() - run_start:.2f}",
        }

    pending_layer_rows = []
    pending_head_rows = []
    pending_group_rows = []

    def clone_for_ablation(record):
        return clone_cache_entry(record["packed_cache"])

    if not score_only_state_norm:
        if "layer" in args.ablation_levels:
            print("Computing layer-level activation_delta sensitivity")
            item_count = 0
            for layer_idx, mixer in selected_mixer_pairs:
                for record in prompt_records:
                    ablated_cache = clone_for_ablation(record)
                    zero_layer_state_(ablated_cache, mixer)
                    logits, loss, ppl = decode_from_packed_cache(
                        model,
                        ablated_cache,
                        record["decode_inputs"],
                        record["continuation_ids"],
                        device,
                        sync_each_token=args.sync_each_token,
                    )
                    decode_call_index += 1
                    diff = summarize_logits_diff(record["baseline_logits"], logits)
                    layer_agg[layer_idx]["loss_increase"] += loss - record["baseline_loss"]
                    layer_agg[layer_idx]["ppl_increase"] += ppl - record["baseline_ppl"]
                    layer_agg[layer_idx]["top1_drop"] += 1.0 - diff["top1_match_rate"]
                    item_count += 1
                    if item_count % args.flush_every == 0:
                        print(f"  layer progress {item_count} items, decode_calls={decode_call_index}/{total_calls}")

        if "head" in args.ablation_levels:
            print("Computing head-level activation_delta sensitivity")
            item_count = 0
            for layer_idx, mixer in selected_mixer_pairs:
                for head_id in probe_heads:
                    key = (layer_idx, head_id)
                    for record in prompt_records:
                        ablated_cache = clone_for_ablation(record)
                        zero_head_state_(ablated_cache, mixer, head_id)
                        _, loss, _ = decode_from_packed_cache(
                            model,
                            ablated_cache,
                            record["decode_inputs"],
                            record["continuation_ids"],
                            device,
                            sync_each_token=args.sync_each_token,
                        )
                        decode_call_index += 1
                        head_agg[key]["loss_increase"] += loss - record["baseline_loss"]
                        item_count += 1
                        if item_count % args.flush_every == 0:
                            print(f"  head progress {item_count} items, decode_calls={decode_call_index}/{total_calls}")

        if "group" in args.ablation_levels:
            print("Computing channel-group activation_delta sensitivity")
            item_count = 0
            for layer_idx, mixer in selected_mixer_pairs:
                for head_id in probe_heads:
                    for group_id in probe_channel_groups:
                        key = (layer_idx, head_id, group_id)
                        for record in prompt_records:
                            ablated_cache = clone_for_ablation(record)
                            zero_group_state_(ablated_cache, mixer, head_id, group_id)
                            _, loss, _ = decode_from_packed_cache(
                                model,
                                ablated_cache,
                                record["decode_inputs"],
                                record["continuation_ids"],
                                device,
                                sync_each_token=args.sync_each_token,
                            )
                            decode_call_index += 1
                            group_agg[key]["loss_increase"] += loss - record["baseline_loss"]
                            item_count += 1
                            if item_count % args.flush_every == 0:
                                print(f"  group progress {item_count} items, decode_calls={decode_call_index}/{total_calls}")

    layer_rows = []
    for layer_idx, stats in layer_agg.items():
        layer_rows.append(
            {
                "layer_idx": layer_idx,
                "state_norm": stats["state_norm"],
                "loss_increase": stats["loss_increase"] / num_prompts,
                "ppl_increase": stats["ppl_increase"] / num_prompts,
                "top1_drop": stats["top1_drop"] / num_prompts,
                **make_row_meta(decode_call_index),
            }
        )

    head_rows = []
    for (layer_idx, head_id), stats in head_agg.items():
        head_rows.append(
            {
                "layer_idx": layer_idx,
                "head_id": head_id,
                "state_norm": stats["state_norm"],
                "loss_increase": stats["loss_increase"] / num_prompts,
                **make_row_meta(decode_call_index),
            }
        )
    head_rows = rank_descending(head_rows, "loss_increase")

    group_rows = []
    for (layer_idx, head_id, group_id), stats in group_agg.items():
        channel_start = group_id * GROUP_SIZE
        channel_end = channel_start + GROUP_SIZE
        group_rows.append(
            {
                "layer_idx": layer_idx,
                "head_id": head_id,
                "channel_group_id": group_id,
                "channel_start": channel_start,
                "channel_end": channel_end,
                "state_norm": stats["state_norm"],
                "loss_increase": stats["loss_increase"] / num_prompts,
                **make_row_meta(decode_call_index),
            }
        )
    group_rows = rank_descending(group_rows, "loss_increase")

    layer_meta = compute_sensitivity_metadata(layer_rows)
    head_meta = compute_sensitivity_metadata(head_rows)
    group_meta = compute_sensitivity_metadata(group_rows)

    for row in layer_rows:
        row.update(layer_meta)
        row["ablation_level"] = "layer"
    for row in head_rows:
        row.update(head_meta)
        row["ablation_level"] = "head"
    for row in group_rows:
        row.update(group_meta)
        row["ablation_level"] = "group"

    if score_only_state_norm:
        head_norm_values = [row["state_norm"] for row in head_rows]
        group_norm_values = [row["state_norm"] for row in group_rows]
        head_loss_values = head_norm_values
        group_loss_values = group_norm_values
    else:
        head_norm_values = [row["state_norm"] for row in head_rows]
        head_loss_values = [row["loss_increase"] for row in head_rows]
        group_norm_values = [row["state_norm"] for row in group_rows]
        group_loss_values = [row["loss_increase"] for row in group_rows]

    head_spearman = spearman_corr(head_norm_values, head_loss_values)
    head_kendall = kendall_tau(head_norm_values, head_loss_values)
    group_spearman = spearman_corr(group_norm_values, group_loss_values)
    group_kendall = kendall_tau(group_norm_values, group_loss_values)

    top_layer_count = max(1, min(len(layer_rows), len(layer_rows)))
    top_head_count = max(1, min(len(head_rows), len(head_rows)))
    top_group_count = max(1, min(len(group_rows), len(group_rows)))

    if score_only_state_norm:
        top_layers_mean = 0.0
        top_heads_mean = 0.0
        top_groups_mean = 0.0
        random_layers_mean = 0.0
        random_heads_mean = 0.0
        random_groups_mean = 0.0
        uniform_layers_mean = 0.0
        uniform_heads_mean = 0.0
        uniform_groups_mean = 0.0
    else:
        top_layers_mean = mean_of_top_by_key(layer_rows, "state_norm", "loss_increase", top_layer_count)
        top_heads_mean = mean_of_top_by_key(head_rows, "state_norm", "loss_increase", top_head_count)
        top_groups_mean = mean_of_top_by_key(group_rows, "state_norm", "loss_increase", top_group_count)
        random_layers_mean = random_mean_of_k(layer_rows, "loss_increase", top_layer_count, args.seed)
        random_heads_mean = random_mean_of_k(head_rows, "loss_increase", top_head_count, args.seed + 1)
        random_groups_mean = random_mean_of_k(group_rows, "loss_increase", top_group_count, args.seed + 2)
        uniform_layers_mean = round_robin_layer_mean(layer_rows, "loss_increase", top_layer_count)
        uniform_heads_mean = round_robin_layer_mean(head_rows, "loss_increase", top_head_count)
        uniform_groups_mean = round_robin_layer_mean(group_rows, "loss_increase", top_group_count)

    head_top_k_share, head_entropy, head_gini = concentration_stats(head_norm_values, top_k=8)
    group_top_k_share, group_entropy, group_gini = concentration_stats(group_norm_values, top_k=32)
    head_rank_stability = average_pairwise_spearman(prompt_head_norm_ranks)
    group_rank_stability = average_pairwise_spearman(prompt_group_norm_ranks)

    if "layer" in args.ablation_levels:
        layer_csv = output_paths["layer"]
        append_rows(
            layer_csv,
            LAYER_FIELDNAMES,
            [
                {
                    **row,
                    "state_norm": f"{row['state_norm']:.8e}",
                    "loss_increase": f"{row['loss_increase']:.8f}",
                    "ppl_increase": f"{row['ppl_increase']:.8f}",
                    "top1_drop": f"{row['top1_drop']:.8f}",
                    "max_loss_increase": f"{float(row['max_loss_increase']):.8f}",
                    "mean_positive_loss_increase": f"{float(row['mean_positive_loss_increase']):.8f}",
                }
                for row in sorted(layer_rows, key=lambda row: row["loss_increase"], reverse=True)
            ],
        )
        copy_latest(layer_csv, output_paths["layer_latest"])

    if "head" in args.ablation_levels:
        head_csv = output_paths["head"]
        append_rows(
            head_csv,
            HEAD_FIELDNAMES,
            [
                {
                    **row,
                    "state_norm": f"{row['state_norm']:.8e}",
                    "loss_increase": f"{row['loss_increase']:.8f}",
                    "max_loss_increase": f"{float(row['max_loss_increase']):.8f}",
                    "mean_positive_loss_increase": f"{float(row['mean_positive_loss_increase']):.8f}",
                }
                for row in head_rows
            ],
        )
        copy_latest(head_csv, output_paths["head_latest"])

    if "group" in args.ablation_levels:
        group_csv = output_paths["group"]
        append_rows(
            group_csv,
            GROUP_FIELDNAMES,
            [
                {
                    **row,
                    "state_norm": f"{row['state_norm']:.8e}",
                    "loss_increase": f"{row['loss_increase']:.8f}",
                    "max_loss_increase": f"{float(row['max_loss_increase']):.8f}",
                    "mean_positive_loss_increase": f"{float(row['mean_positive_loss_increase']):.8f}",
                }
                for row in group_rows
            ],
        )
        copy_latest(group_csv, output_paths["group_latest"])

    summary_rows = [
        {
            "scope": "layer",
            "enabled_ablation_levels": ",".join(args.ablation_levels),
            "top_k_share": f"{head_top_k_share:.8f}",
            "entropy": f"{head_entropy:.8f}",
            "gini": f"{head_gini:.8f}",
            "rank_stability": f"{head_rank_stability:.8f}",
            "spearman_state_norm_vs_loss": f"{head_spearman:.8f}",
            "kendall_state_norm_vs_loss": f"{head_kendall:.8f}",
            "random_ablation_loss_increase": f"{random_layers_mean:.8f}",
            "uniform_ablation_loss_increase": f"{uniform_layers_mean:.8f}",
            "top_sensitive_loss_increase": f"{top_layers_mean:.8f}",
        },
        {
            "scope": "head",
            "enabled_ablation_levels": ",".join(args.ablation_levels),
            "top_k_share": f"{head_top_k_share:.8f}",
            "entropy": f"{head_entropy:.8f}",
            "gini": f"{head_gini:.8f}",
            "rank_stability": f"{head_rank_stability:.8f}",
            "spearman_state_norm_vs_loss": f"{head_spearman:.8f}",
            "kendall_state_norm_vs_loss": f"{head_kendall:.8f}",
            "random_ablation_loss_increase": f"{random_heads_mean:.8f}",
            "uniform_ablation_loss_increase": f"{uniform_heads_mean:.8f}",
            "top_sensitive_loss_increase": f"{top_heads_mean:.8f}",
        },
        {
            "scope": "channel_group",
            "enabled_ablation_levels": ",".join(args.ablation_levels),
            "top_k_share": f"{group_top_k_share:.8f}",
            "entropy": f"{group_entropy:.8f}",
            "gini": f"{group_gini:.8f}",
            "rank_stability": f"{group_rank_stability:.8f}",
            "spearman_state_norm_vs_loss": f"{group_spearman:.8f}",
            "kendall_state_norm_vs_loss": f"{group_kendall:.8f}",
            "random_ablation_loss_increase": f"{random_groups_mean:.8f}",
            "uniform_ablation_loss_increase": f"{uniform_groups_mean:.8f}",
            "top_sensitive_loss_increase": f"{top_groups_mean:.8f}",
        },
    ]
    append_rows(summary_csv, SUMMARY_FIELDNAMES, summary_rows)
    copy_latest(summary_csv, output_paths["summary_latest"])

    if "layer" in args.ablation_levels:
        print(f"Saved layer sensitivity to {output_paths['layer']}")
        print(f"Updated latest layer sensitivity to {output_paths['layer_latest']}")
    if "head" in args.ablation_levels:
        print(f"Saved head sensitivity to {output_paths['head']}")
        print(f"Updated latest head sensitivity to {output_paths['head_latest']}")
    if "group" in args.ablation_levels:
        print(f"Saved channel-group sensitivity to {output_paths['group']}")
        print(f"Updated latest group sensitivity to {output_paths['group_latest']}")
    print(f"Saved summary to {summary_csv}")

    top_layers = sorted(layer_rows, key=lambda row: row["loss_increase"], reverse=True)[:8]
    print("Top sensitive layers")
    for row in top_layers:
        print(
            f"  layer={row['layer_idx']:02d} loss_increase={row['loss_increase']:.8f} "
            f"ppl_increase={row['ppl_increase']:.8f} top1_drop={row['top1_drop']:.8f}"
        )

    print("Top sensitive heads")
    for row in head_rows[:8]:
        print(
            f"  layer={row['layer_idx']:02d} head={row['head_id']:02d} "
            f"state_norm={row['state_norm']:.8e} loss_increase={row['loss_increase']:.8f}"
        )

    print("Top sensitive channel groups")
    for row in group_rows[:8]:
        print(
            f"  layer={row['layer_idx']:02d} head={row['head_id']:02d} group={row['channel_group_id']:02d} "
            f"channels=[{row['channel_start']},{row['channel_end']}) state_norm={row['state_norm']:.8e} "
            f"loss_increase={row['loss_increase']:.8f}"
        )

    print(
        f"Head correlation: spearman={head_spearman:.8f} kendall={head_kendall:.8f}; "
        f"Group correlation: spearman={group_spearman:.8f} kendall={group_kendall:.8f}"
    )
    print(
        f"Top-sensitive group mean loss increase={top_groups_mean:.8f}, "
        f"random baseline={random_groups_mean:.8f}, uniform baseline={uniform_groups_mean:.8f}"
    )

    if not score_only_state_norm:
        print(
            f"Top-by-state-norm means: layer={top_layers_mean:.8f}, head={top_heads_mean:.8f}, group={top_groups_mean:.8f}"
        )
        print(
            f"Random baseline means  : layer={random_layers_mean:.8f}, head={random_heads_mean:.8f}, group={random_groups_mean:.8f}"
        )
        print(
            f"Uniform baseline means : layer={uniform_layers_mean:.8f}, head={uniform_heads_mean:.8f}, group={uniform_groups_mean:.8f}"
        )
        if group_top_k_share > 0.25 and top_groups_mean > max(random_groups_mean, uniform_groups_mean) * 1.5:
            print("[PASS] State-channel importance is non-uniform and cache management is promising.")
        else:
            print("[WARN] State-channel importance is weak; need better scoring or task-specific signals.")

    if args.fast_smoke:
        print("[PASS] fast_smoke completed")


if __name__ == "__main__":
    main()
