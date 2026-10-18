import argparse
import copy
import csv
import hashlib
import math
import random
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mamba2_permutation_utils import (
    bad_permute_mamba2_inproj_only_,
    make_head_index,
    permute_mamba2_heads_,
)
from output_naming import build_output_path
from phase6_eval_prefix_quality import sample_validation_texts
from phase12_mamba2_head_permutation_equivalence import (
    MODEL_NAME,
    TOKENIZER_NAME,
    build_model,
    collect_mamba2_mixers,
    configure_mamba2_for_equivalence,
    ensure_local_hf_cache,
    summarize_logits_diff,
)
from phase17_mamba2_intra_head_channel_permutation_probe import (
    bad_permute_mamba2_intra_head_inproj_only_,
    make_intra_head_channel_perm,
    permute_mamba2_intra_head_channels_,
)


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


GOOD_REL_THRESHOLD = 2e-4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perm_type", nargs="+", choices=["head", "intra_head"], default=["head", "intra_head"])
    parser.add_argument("--layers", type=str, default="1")
    parser.add_argument("--layer_ids", type=str, default=None)
    parser.add_argument("--heads", type=str, default="all")
    parser.add_argument("--prefix_len", type=int, default=128)
    parser.add_argument("--continuation_len", type=int, default=32)
    parser.add_argument("--num_prompts", type=int, default=8)
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=["good_weight_and_state_perm", "bad_weight_only_perm", "bad_state_only_perm"],
        default=["good_weight_and_state_perm", "bad_weight_only_perm", "bad_state_only_perm"],
    )
    parser.add_argument("--disable_mem_eff_path", action="store_true")
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "synthetic"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--debug_layers", action="store_true")
    parser.add_argument("--verbose_perm", action="store_true")
    parser.add_argument("--single_layer_debug", action="store_true")
    parser.add_argument("--append_each_prompt", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--smoke_no_model", action="store_true")
    parser.add_argument("--dry_run_samples", action="store_true")
    return parser.parse_args()


def synchronize_if_cuda(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def parse_layer_spec(layers_arg: str, total_layers: int):
    if layers_arg == "all":
        return list(range(total_layers))
    if layers_arg.startswith("first:"):
        count = int(layers_arg.split(":", 1)[1])
        if count < 1 or count > total_layers:
            raise ValueError(f"Invalid first:N layer spec: {layers_arg}")
        return list(range(count))
    if "," in layers_arg:
        layer_indices = sorted({int(token.strip()) for token in layers_arg.split(",") if token.strip()})
    else:
        layer_idx = int(layers_arg)
        layer_indices = [layer_idx]
    invalid = [layer_idx for layer_idx in layer_indices if layer_idx < 0 or layer_idx >= total_layers]
    if invalid:
        raise ValueError(f"Invalid layer indices: {invalid}")
    return layer_indices


def parse_layer_ids_spec(layer_ids_arg: str | None, total_layers: int):
    if layer_ids_arg is None:
        return None
    if layer_ids_arg == "all":
        return list(range(total_layers))
    layer_indices = sorted({int(token.strip()) for token in layer_ids_arg.split(",") if token.strip()})
    invalid = [layer_idx for layer_idx in layer_indices if layer_idx < 0 or layer_idx >= total_layers]
    if invalid:
        raise ValueError(f"Invalid layer ids: {invalid}")
    return layer_indices


def parse_head_spec(heads_arg: str, nheads: int):
    if heads_arg == "all":
        return list(range(nheads))
    head_indices = sorted({int(token.strip()) for token in heads_arg.split(",") if token.strip()})
    invalid = [head_id for head_id in head_indices if head_id < 0 or head_id >= nheads]
    if invalid:
        raise ValueError(f"Invalid head indices: {invalid}")
    return head_indices


def sample_token_chunks(tokenizer, dataset_name: str, split: str, num_prompts: int, prefix_len: int, continuation_len: int, seed: int):
    total_len = prefix_len + continuation_len
    if dataset_name == "synthetic":
        vocab_size = int(tokenizer.vocab_size or getattr(tokenizer, "_tokenizer", None).get_vocab_size() if getattr(tokenizer, "_tokenizer", None) is not None else 50257)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        prompts = []
        for _ in range(num_prompts):
            ids = torch.randint(low=0, high=vocab_size, size=(total_len,), generator=generator, dtype=torch.long)
            prompts.append(ids)
        print(f"[sample] dataset={dataset_name} split={split} total_token_count=synthetic requested_total_len={total_len} generated_sample_count={len(prompts)} first_3_start_offsets=[synthetic]")
        return prompts

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
    print(f"[sample] dataset={dataset_name} split={split} total_token_count={total_token_count} requested_total_len={total_len}")

    if total_token_count < total_len + 1:
        raise RuntimeError(f"Insufficient tokens: total_token_count={total_token_count}, required_at_least={total_len + 1}")

    max_start = total_token_count - total_len
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if max_start > 0:
        starts = torch.randint(low=0, high=max_start, size=(num_prompts,), generator=generator).tolist()
    else:
        starts = [0] * num_prompts

    prompts = []
    for start in starts:
        input_ids = all_ids[start:start + total_len]
        if input_ids.numel() != total_len:
            continue
        prompts.append(input_ids.clone())

    if len(prompts) < num_prompts:
        prompts = []
        if num_prompts <= 1:
            fallback_starts = [0]
        else:
            fallback_max = max(0, max_start)
            if fallback_max == 0:
                fallback_starts = [0] * num_prompts
            else:
                fallback_starts = torch.linspace(0, fallback_max, steps=num_prompts).round().long().tolist()
        for start in fallback_starts:
            input_ids = all_ids[start:start + total_len]
            if input_ids.numel() == total_len:
                prompts.append(input_ids.clone())
            if len(prompts) >= num_prompts:
                break

    prompts = prompts[:num_prompts]
    first_starts = starts[:3] if starts else []
    print(f"[sample] generated_sample_count={len(prompts)} first_3_start_offsets={first_starts}")
    return prompts


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
        prefix_ids = input_ids[:, :prefix_len]
        continuation_ids = input_ids[:, prefix_len:]
        samples.append(
            {
                "prompt_id": prompt_id,
                "dataset": "synthetic",
                "input_ids": input_ids,
                "prefix_ids": prefix_ids,
                "continuation_ids": continuation_ids,
            }
        )
    print(f"[sample] synthetic sample count={len(samples)} prefix_len={prefix_len} continuation_len={continuation_len} total_len={total_len}")
    if samples:
        print(f"[sample] first synthetic sample token shape={tuple(samples[0]['input_ids'].shape)}")
    return samples


def normalize_eval_sample(sample_entry):
    prompt_id, prefix_ids, continuation_ids = sample_entry
    if prefix_ids.dim() > 1:
        prefix_ids = prefix_ids.reshape(-1)
    if continuation_ids.dim() > 1:
        continuation_ids = continuation_ids.reshape(-1)
    return prompt_id, prefix_ids, continuation_ids


def clone_cache_entry(entry):
    if torch.is_tensor(entry):
        return entry.clone()
    if isinstance(entry, tuple):
        return tuple(clone_cache_entry(item) for item in entry)
    if isinstance(entry, list):
        return [clone_cache_entry(item) for item in entry]
    if isinstance(entry, dict):
        return {key: clone_cache_entry(value) for key, value in entry.items()}
    return copy.deepcopy(entry)


def clone_inference_params(inference_params):
    from mamba_ssm.utils.generation import InferenceParams

    cloned = InferenceParams(
        max_seqlen=inference_params.max_seqlen,
        max_batch_size=inference_params.max_batch_size,
        seqlen_offset=inference_params.seqlen_offset,
        batch_size_offset=inference_params.batch_size_offset,
        key_value_memory_dict={},
        lengths_per_sample=(None if inference_params.lengths_per_sample is None else inference_params.lengths_per_sample.clone()),
    )
    cloned.key_value_memory_dict = {
        layer_idx: clone_cache_entry(cache_entry)
        for layer_idx, cache_entry in inference_params.key_value_memory_dict.items()
    }
    return cloned


def build_head_perm(nheads: int, selected_heads, seed: int, device):
    full_perm = torch.arange(nheads, device=device, dtype=torch.long)
    if len(selected_heads) == nheads:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return torch.randperm(nheads, generator=generator, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shuffled = torch.tensor(selected_heads, device=device, dtype=torch.long)[
        torch.randperm(len(selected_heads), generator=generator, device=device)
    ]
    full_perm[torch.tensor(selected_heads, device=device, dtype=torch.long)] = shuffled
    return full_perm


def build_perm_specs(mixers, perm_type: str, target_layers, target_heads, seed: int, debug: bool = False):
    perm_specs = {}
    build_errors = {}
    mixer_lookup = {layer_idx: mixer for layer_idx, mixer in mixers}
    for layer_idx in target_layers:
        mixer = mixer_lookup[layer_idx]
        try:
            if perm_type == "head":
                perm = build_head_perm(int(mixer.nheads), target_heads, seed + layer_idx * 10007, mixer.in_proj.weight.device)
                perm_specs[layer_idx] = {"perm_type": "head", "perm": perm, "nheads": int(mixer.nheads), "headdim": int(mixer.headdim)}
            else:
                head_perms = {
                    head_id: make_intra_head_channel_perm(mixer, head_id=head_id, seed=seed + layer_idx * 10007)
                    for head_id in target_heads
                }
                perm_specs[layer_idx] = {
                    "perm_type": "intra_head",
                    "head_perms": head_perms,
                    "nheads": int(mixer.nheads),
                    "headdim": int(mixer.headdim),
                    "d_inner": int(mixer.d_inner),
                    "d_state": int(mixer.d_state),
                }
        except Exception as exc:
            tb = traceback.format_exc()
            error_msg = f"layer_idx={layer_idx} perm_type={perm_type} build_perm_spec_failed: {exc} | {tb.strip()}"
            build_errors[layer_idx] = error_msg
            if debug:
                print(f"[ERROR] {error_msg}")
            perm_specs[layer_idx] = {
                "perm_type": perm_type,
                "error": error_msg,
                "nheads": int(mixer.nheads),
                "headdim": int(mixer.headdim),
                "d_inner": int(mixer.d_inner),
                "d_state": int(mixer.d_state),
            }
    return perm_specs, build_errors


def apply_weight_permutation(model, perm_type: str, target_layers, perm_specs, bad: bool = False):
    for layer_idx, mixer in collect_mamba2_mixers(model):
        if layer_idx not in target_layers:
            continue
        perm_spec = perm_specs[layer_idx]
        if perm_type == "head":
            if bad:
                bad_permute_mamba2_inproj_only_(mixer, perm_spec["perm"])
            else:
                permute_mamba2_heads_(mixer, perm_spec["perm"])
        else:
            for head_id, perm in perm_spec["head_perms"].items():
                if bad:
                    bad_permute_mamba2_intra_head_inproj_only_(mixer, head_id, perm)
                else:
                    permute_mamba2_intra_head_channels_(mixer, head_id, perm)


def _permute_axis_segment_(tensor: torch.Tensor, axis: int, start: int, end: int, index: torch.Tensor):
    selection = [slice(None)] * tensor.dim()
    selection[axis] = slice(start, end)
    segment = tensor[tuple(selection)].clone()
    permuted = segment.index_select(axis, index.to(segment.device))
    tensor[tuple(selection)] = permuted


def _permute_axis_full_(tensor: torch.Tensor, axis: int, index: torch.Tensor):
    return tensor.index_select(axis, index.to(tensor.device))


def _permute_intra_head_axis_(tensor: torch.Tensor, axis: int, head_axis: int, head_id: int, perm: torch.Tensor):
    index = [slice(None)] * tensor.dim()
    index[head_axis] = head_id
    sub_tensor = tensor[tuple(index)].clone()
    permuted = sub_tensor.index_select(axis - (1 if axis > head_axis else 0), perm.to(sub_tensor.device))
    tensor[tuple(index)] = permuted


def _permute_tensor_by_spec_(tensor: torch.Tensor, tensor_name: str, perm_spec: dict, debug: bool = False):
    shape = tuple(tensor.shape)
    nheads = perm_spec["nheads"]
    headdim = perm_spec["headdim"]
    d_inner = perm_spec.get("d_inner", nheads * headdim)
    d_plus_bc = d_inner + 2 * perm_spec.get("d_state", 128)

    if debug:
        print(f"[debug] cache tensor={tensor_name} shape={shape} perm_type={perm_spec['perm_type']}")

    if tensor.dim() >= 4 and nheads in shape and headdim in shape:
        head_axis = next(axis for axis, size in enumerate(shape) if size == nheads)
        headdim_axis = next(axis for axis, size in enumerate(shape) if size == headdim and axis != head_axis)
        if perm_spec["perm_type"] == "head":
            print(f"cache tensor={tensor_name} shape={shape} permuted_axis={head_axis} mode=head")
            return _permute_axis_full_(tensor, head_axis, perm_spec["perm"])
        for head_id, perm in perm_spec["head_perms"].items():
            print(f"cache tensor={tensor_name} shape={shape} permuted_axis={headdim_axis} mode=intra_head head={head_id}")
            _permute_intra_head_axis_(tensor, headdim_axis, head_axis, head_id, perm)
        return tensor

    for axis, size in enumerate(shape):
        if size == d_plus_bc:
            if perm_spec["perm_type"] == "head":
                head_index = make_head_index(perm_spec["perm"], headdim)
                print(f"cache tensor={tensor_name} shape={shape} permuted_axis={axis} mode=head_xbc")
                _permute_axis_segment_(tensor, axis, 0, d_inner, head_index)
            else:
                for head_id, perm in perm_spec["head_perms"].items():
                    c0 = head_id * headdim
                    c1 = (head_id + 1) * headdim
                    print(f"cache tensor={tensor_name} shape={shape} permuted_axis={axis} mode=intra_head_xbc head={head_id}")
                    _permute_axis_segment_(tensor, axis, c0, c1, perm)
            return tensor
        if size == d_inner:
            if perm_spec["perm_type"] == "head":
                head_index = make_head_index(perm_spec["perm"], headdim)
                print(f"cache tensor={tensor_name} shape={shape} permuted_axis={axis} mode=head_x")
                return _permute_axis_full_(tensor, axis, head_index)
            for head_id, perm in perm_spec["head_perms"].items():
                c0 = head_id * headdim
                c1 = (head_id + 1) * headdim
                print(f"cache tensor={tensor_name} shape={shape} permuted_axis={axis} mode=intra_head_x head={head_id}")
                _permute_axis_segment_(tensor, axis, c0, c1, perm)
            return tensor

    for axis, size in enumerate(shape):
        if size == nheads and perm_spec["perm_type"] == "head":
            print(f"cache tensor={tensor_name} shape={shape} permuted_axis={axis} mode=head_axis")
            return _permute_axis_full_(tensor, axis, perm_spec["perm"])
    print(f"[WARN] unrecognized cache tensor shape for {tensor_name}: shape={shape} perm_type={perm_spec['perm_type']}")
    return tensor


def permute_mamba2_cache_(cache, layer_idx: int, perm_spec: dict, debug: bool = False):
    warnings = []
    if layer_idx not in cache:
        return warnings
    layer_cache = cache[layer_idx]
    if isinstance(layer_cache, tuple):
        names = ["conv_state", "ssm_state"]
        new_items = []
        for item_name, item in zip(names, layer_cache):
            if torch.is_tensor(item):
                new_items.append(_permute_tensor_by_spec_(item, f"layer{layer_idx}.{item_name}", perm_spec, debug=debug))
            else:
                new_items.append(item)
        cache[layer_idx] = tuple(new_items)
    elif isinstance(layer_cache, dict):
        for item_name, item in layer_cache.items():
            if torch.is_tensor(item):
                layer_cache[item_name] = _permute_tensor_by_spec_(item, f"layer{layer_idx}.{item_name}", perm_spec, debug=debug)
    elif isinstance(layer_cache, list):
        for item_idx, item in enumerate(layer_cache):
            if torch.is_tensor(item):
                layer_cache[item_idx] = _permute_tensor_by_spec_(item, f"layer{layer_idx}.item{item_idx}", perm_spec, debug=debug)
    elif torch.is_tensor(layer_cache):
        cache[layer_idx] = _permute_tensor_by_spec_(layer_cache, f"layer{layer_idx}.tensor", perm_spec, debug=debug)
    if "error" in perm_spec:
        warnings.append(perm_spec["error"])
    return warnings


def write_row(output_path: Path, csv_fieldnames, row):
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writerow(row)


def make_error_row(args, perm_type, mode, prompt_id, target_layers, target_heads, error_msg):
    return {
        "perm_type": perm_type,
        "mode": mode,
        "prompt_id": prompt_id,
        "layers": args.layers,
        "layer_ids": args.layer_ids if getattr(args, "layer_ids", None) is not None else "",
        "heads": args.heads,
        "target_layers": str(target_layers),
        "target_heads": str(target_heads),
        "prefix_len": args.prefix_len,
        "continuation_len": args.continuation_len,
        "disable_mem_eff_path": args.disable_mem_eff_path,
        "max_abs_diff": "",
        "mean_abs_diff": "",
        "relative_max_error": "",
        "top1_match_rate": "",
        "continuation_loss": "",
        "continuation_ppl": "",
        "status": "error",
        "error_msg": error_msg,
    }


def make_started_row(args, perm_type, prompt_id, target_layers, target_heads):
    return {
        "perm_type": perm_type,
        "mode": "started",
        "prompt_id": prompt_id,
        "layers": args.layers,
        "layer_ids": args.layer_ids if getattr(args, "layer_ids", None) is not None else "",
        "heads": args.heads,
        "target_layers": str(target_layers),
        "target_heads": str(target_heads),
        "prefix_len": args.prefix_len,
        "continuation_len": args.continuation_len,
        "disable_mem_eff_path": args.disable_mem_eff_path,
        "max_abs_diff": "",
        "mean_abs_diff": "",
        "relative_max_error": "",
        "top1_match_rate": "",
        "continuation_loss": "",
        "continuation_ppl": "",
        "status": "started",
        "error_msg": "",
    }


def log_startup_state(args, target_layers, target_heads, modes, num_prompts, output_path):
    print(f"parsed args: {args}")
    print(f"target_layers: {target_layers}")
    print(f"target_heads: {target_heads}")
    print(f"modes: {modes}")
    print(f"num_prompts: {num_prompts}")
    print(f"output_csv path: {output_path}")


def build_inference_params(max_seqlen: int, batch_size: int):
    from mamba_ssm.utils.generation import InferenceParams

    return InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)


@torch.no_grad()
def prefill_prefix(model, prefix_ids: torch.Tensor, max_seqlen: int, device: str):
    inference_params = build_inference_params(max_seqlen=max_seqlen, batch_size=1)
    batch = prefix_ids.unsqueeze(0).to(device)
    synchronize_if_cuda(device)
    outputs = model(batch, inference_params=inference_params, num_last_tokens=1)
    synchronize_if_cuda(device)
    inference_params.seqlen_offset = prefix_ids.numel()
    return outputs.logits.float().cpu(), inference_params


@torch.no_grad()
def decode_continuation(model, inference_params, prefill_last_logits, continuation_ids: torch.Tensor, device: str):
    logits_chunks = [prefill_last_logits]
    for token_idx in range(continuation_ids.numel() - 1):
        token = continuation_ids[token_idx].view(1, 1).to(device)
        synchronize_if_cuda(device)
        outputs = model(token, inference_params=inference_params, num_last_tokens=1)
        synchronize_if_cuda(device)
        logits_chunks.append(outputs.logits.float().cpu())
        inference_params.seqlen_offset += 1
    return torch.cat(logits_chunks, dim=1)


def continuation_metrics(logits: torch.Tensor, continuation_ids: torch.Tensor):
    targets = continuation_ids.unsqueeze(0)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean").item()
    return loss, math.exp(loss)


def classify_status(mode: str, metrics: dict, reference_good_metrics: dict | None):
    if mode == "good_weight_and_state_perm":
        if metrics["top1_match_rate"] >= 1.0 - 1e-8 and metrics["relative_max_error"] < GOOD_REL_THRESHOLD:
            return "functional_pass"
        return "fail"
    if reference_good_metrics is None:
        return "unknown"
    max_worse = metrics["max_abs_diff"] > max(reference_good_metrics["max_abs_diff"] * 10.0, 1e-2)
    top1_drop = metrics["top1_match_rate"] < reference_good_metrics["top1_match_rate"] - 1e-3
    return "expected_fail" if (max_worse or top1_drop) else "weak_fail"


def run_mode_for_sample(reference_model, candidate_model, prefix_ids, continuation_ids, max_seqlen, device: str):
    prefill_last_logits, reference_cache = prefill_prefix(reference_model, prefix_ids, max_seqlen, device)
    reference_logits = decode_continuation(
        reference_model,
        clone_inference_params(reference_cache),
        prefill_last_logits,
        continuation_ids,
        device,
    )

    candidate_prefill_logits, candidate_cache = prefill_prefix(candidate_model, prefix_ids, max_seqlen, device)
    candidate_logits = decode_continuation(candidate_model, candidate_cache, candidate_prefill_logits, continuation_ids, device)
    diff_metrics = summarize_logits_diff(reference_logits, candidate_logits)
    loss, ppl = continuation_metrics(candidate_logits, continuation_ids)
    diff_metrics["continuation_loss"] = loss
    diff_metrics["continuation_ppl"] = ppl
    return diff_metrics, reference_cache


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.dry_run_samples:
        print(f"[dry_run_samples] dataset={args.dataset}")
        print(f"[dry_run_samples] prefix_len={args.prefix_len}")
        print(f"[dry_run_samples] continuation_len={args.continuation_len}")
        print(f"[dry_run_samples] num_prompts={args.num_prompts}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        if args.dataset == "synthetic":
            samples = build_synthetic_eval_samples(
                tokenizer,
                num_prompts=args.num_prompts,
                prefix_len=args.prefix_len,
                continuation_len=args.continuation_len,
                seed=args.seed,
            )
        elif args.dataset == "wikitext2":
            samples = sample_token_chunks(
                tokenizer,
                dataset_name=args.dataset,
                split="validation",
                num_prompts=args.num_prompts,
                prefix_len=args.prefix_len,
                continuation_len=args.continuation_len,
                seed=args.seed,
            )
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        total_len = args.prefix_len + args.continuation_len
        print(f"total token count: {'synthetic' if args.dataset == 'synthetic' else 'see above'}")
        print(f"requested total_len: {total_len}")
        print(f"generated sample count: {len(samples)}")
        first_shapes = []
        for sample in samples[:3]:
            if isinstance(sample, dict):
                first_shapes.append(tuple(sample["input_ids"].shape))
            else:
                first_shapes.append(tuple(sample.shape))
        print(f"first 3 sample shapes: {first_shapes}")
        return

    config_blob = "|".join(
        [
            f"dataset={args.dataset}",
            f"perm_type={'-'.join(args.perm_type)}",
            f"layers={args.layers}",
            f"layer_ids={args.layer_ids or ''}",
            f"heads={args.heads}",
            f"modes={'-'.join(args.mode)}",
            f"prefix_len={args.prefix_len}",
            f"continuation_len={args.continuation_len}",
            f"num_prompts={args.num_prompts}",
            f"seed={args.seed}",
        ]
    )
    config_digest = hashlib.sha1(config_blob.encode("utf-8")).hexdigest()[:12]
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        filename = f"phase21_mamba2_state_cache_permutation_equivalence__{config_digest}.csv"
        output_path = Path("outputs") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_local_hf_cache(output_path.parent)

    print("Cache path: inference_params.key_value_memory_dict")
    print("Checked API: allocate_inference_cache=True, inference_params=True, layer cache=(conv_state, ssm_state)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    rows = []
    summary_by_mode = defaultdict(list)
    target_layers = []
    target_heads = []
    prompt_loop_executed = False
    rows_written = {"count": 0}
    csv_fieldnames = [
        "perm_type",
        "mode",
        "prompt_id",
        "layers",
        "layer_ids",
        "heads",
        "target_layers",
        "target_heads",
        "prefix_len",
        "continuation_len",
        "disable_mem_eff_path",
        "max_abs_diff",
        "mean_abs_diff",
        "relative_max_error",
        "top1_match_rate",
        "continuation_loss",
        "continuation_ppl",
        "status",
        "error_msg",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()

    def append_row(row):
        write_row(output_path, csv_fieldnames, row)
        rows.append(row)
        rows_written["count"] += 1

    def append_single_error(perm_type, mode, prompt_id, err_target_layers, err_target_heads, error_msg):
        append_row(make_error_row(args, perm_type, mode, prompt_id, err_target_layers, err_target_heads, error_msg))

    def append_error_rows(perm_type, prompt_id, error_msg):
        error_modes = args.mode or ["error"]
        for mode in error_modes:
            append_single_error(perm_type, mode, prompt_id, target_layers, target_heads, error_msg)

    if not args.mode:
        log_startup_state(args, target_layers, target_heads, args.mode, args.num_prompts, output_path)
        append_single_error("preflight", "error", -1, target_layers, target_heads, "No modes selected")
        print(f"Saved results to {output_path}")
        return

    if args.smoke_no_model:
        smoke_target_layers = []
        if args.layer_ids is not None:
            smoke_target_layers = [int(token.strip()) for token in args.layer_ids.split(",") if token.strip()]
        elif args.layers == "all":
            smoke_target_layers = [0]
        elif "," in args.layers:
            smoke_target_layers = [int(token.strip()) for token in args.layers.split(",") if token.strip()]
        elif args.layers.strip():
            smoke_target_layers = [int(args.layers)]
        smoke_target_heads = [0] if args.heads == "all" else [int(token.strip()) for token in args.heads.split(",") if token.strip()]
        log_startup_state(args, smoke_target_layers, smoke_target_heads, args.mode, args.num_prompts, output_path)
        if not smoke_target_layers:
            append_single_error("smoke", "error", -1, smoke_target_layers, smoke_target_heads, "target_layers is empty in smoke_no_model")
        else:
            started_row = make_started_row(args, args.perm_type[0] if args.perm_type else "smoke", 0, smoke_target_layers, smoke_target_heads)
            append_row(started_row)
            append_row({**started_row, "mode": args.mode[0], "status": "functional_pass", "error_msg": "smoke_no_model"})
        if rows_written["count"] == 0:
            append_single_error("smoke", "error", -1, smoke_target_layers, smoke_target_heads, "smoke_no_model produced no rows")
        print(f"Saved results to {output_path}")
        return

    try:
        print("Cache path: inference_params.key_value_memory_dict")
        print("Checked API: allocate_inference_cache=True, inference_params=True, layer cache=(conv_state, ssm_state)")

        print(f"Loading tokenizer: {TOKENIZER_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        print(f"Loading model: {MODEL_NAME}")
        base_model, loader_name = build_model(device=device, dtype=dtype)
        print(f"Model loader: {loader_name}")
        base_model = configure_mamba2_for_equivalence(base_model, disable_mem_eff_path=args.disable_mem_eff_path)

        mixers = collect_mamba2_mixers(base_model)
        if not mixers:
            raise RuntimeError("No Mamba2 mixer layers found")
        total_layers = len(mixers)
        target_layers = parse_layer_ids_spec(args.layer_ids, total_layers)
        if target_layers is None:
            target_layers = parse_layer_spec(args.layers, total_layers)
        target_heads = parse_head_spec(args.heads, int(mixers[0][1].nheads))
        total_len = args.prefix_len + args.continuation_len
        log_startup_state(args, target_layers, target_heads, args.mode, args.num_prompts, output_path)

        if not target_layers:
            append_single_error("preflight", "error", -1, target_layers, target_heads, "target_layers is empty")
            print(f"Saved results to {output_path}")
            return
        if not target_heads:
            append_single_error("preflight", "error", -1, target_layers, target_heads, "target_heads is empty")
            print(f"Saved results to {output_path}")
            return

        eval_samples = []
        print(f"Loading and tokenizing dataset={args.dataset}")
        if args.dataset == "synthetic":
            synthetic_samples = build_synthetic_eval_samples(
                tokenizer,
                num_prompts=args.num_prompts,
                prefix_len=args.prefix_len,
                continuation_len=args.continuation_len,
                seed=args.seed,
            )
            eval_samples = [
                normalize_eval_sample((sample["prompt_id"], sample["prefix_ids"], sample["continuation_ids"]))
                for sample in synthetic_samples
            ]
        elif args.dataset == "wikitext2":
            samples = sample_token_chunks(
                tokenizer,
                dataset_name=args.dataset,
                split="validation",
                num_prompts=args.num_prompts,
                prefix_len=args.prefix_len,
                continuation_len=args.continuation_len,
                seed=args.seed,
            )
            eval_samples = [
                normalize_eval_sample((prompt_id, input_ids[: args.prefix_len].reshape(-1), input_ids[args.prefix_len : total_len].reshape(-1)))
                for prompt_id, input_ids in enumerate(samples)
                if input_ids.numel() == total_len
            ]
            print(f"[sample] eval_sample_count={len(eval_samples)}")
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        if len(eval_samples) == 0:
            append_single_error("preflight", "error", -1, target_layers, target_heads, "no_valid_samples")
            print(f"Saved results to {output_path}")
            return

        layer_groups = [[layer_idx] for layer_idx in target_layers] if args.single_layer_debug else [target_layers]
        if args.single_layer_debug:
            print(f"[debug] single_layer_debug enabled; processing {len(layer_groups)} layer groups one-by-one")
        if args.append_each_prompt:
            print("[debug] append_each_prompt enabled; rows are flushed immediately after each prompt")

        for perm_type in args.perm_type:
            print(f"\n=== perm_type={perm_type} ===")
            for layer_group in layer_groups:
                group_target_layers = list(layer_group)
                print(f"--- layer_group={group_target_layers} ---")
                perm_specs, build_errors = build_perm_specs(mixers, perm_type, group_target_layers, target_heads, args.seed, debug=args.debug_layers)
                if args.debug_layers:
                    print(f"[debug] selected layer ids: {group_target_layers}")
                    print(f"[debug] selected head ids: {target_heads}")
                    for layer_idx in group_target_layers:
                        print(f"[debug] layer {layer_idx} perm spec: {perm_specs.get(layer_idx)}")
                if build_errors and not args.single_layer_debug:
                    error_msg = " ; ".join(build_errors[layer_idx] for layer_idx in sorted(build_errors))
                    print(f"[ERROR] perm_type={perm_type} build_perm_specs failed for layers={sorted(build_errors)}")
                    print(error_msg)
                    append_error_rows(perm_type, prompt_id=-1, error_msg=error_msg)
                    if args.stop_on_error:
                        raise RuntimeError(error_msg)
                    continue

                good_weight_model = copy.deepcopy(base_model).eval()
                apply_weight_permutation(good_weight_model, perm_type, group_target_layers, perm_specs, bad=False)
                bad_weight_model = copy.deepcopy(base_model).eval()
                apply_weight_permutation(bad_weight_model, perm_type, group_target_layers, perm_specs, bad=True)

                for prompt_id, prefix_ids, continuation_ids in eval_samples:
                    prompt_loop_executed = True
                    print(f"prompt_id={prompt_id} prefix_len={prefix_ids.numel()} continuation_len={continuation_ids.numel()}")
                    append_row(make_started_row(args, perm_type, prompt_id, group_target_layers, target_heads))
                    try:
                        reference_prefill_last_logits, reference_cache = prefill_prefix(base_model, prefix_ids, total_len, device)
                        reference_logits = decode_continuation(
                            base_model,
                            clone_inference_params(reference_cache),
                            reference_prefill_last_logits,
                            continuation_ids,
                            device,
                        )
                        good_reference_cache = clone_inference_params(reference_cache)
                        cache_warnings = []
                        for layer_idx in group_target_layers:
                            cache_warnings.extend(
                                permute_mamba2_cache_(good_reference_cache.key_value_memory_dict, layer_idx, perm_specs[layer_idx], debug=args.debug_layers)
                            )

                        good_prefill_last_logits = reference_prefill_last_logits
                        good_logits = decode_continuation(
                            good_weight_model,
                            good_reference_cache,
                            good_prefill_last_logits,
                            continuation_ids,
                            device,
                        )
                        good_metrics = summarize_logits_diff(reference_logits, good_logits)
                        good_loss, good_ppl = continuation_metrics(good_logits, continuation_ids)
                        good_metrics["continuation_loss"] = good_loss
                        good_metrics["continuation_ppl"] = good_ppl
                        if cache_warnings:
                            warning_text = " ; ".join(cache_warnings)
                            print(f"[WARN] permute_mamba2_cache_ warnings: {warning_text}")
                            good_metrics["warning_msg"] = warning_text

                        for mode in args.mode:
                            try:
                                if mode == "good_weight_and_state_perm":
                                    metrics = dict(good_metrics)
                                elif mode == "bad_weight_only_perm":
                                    bad_weight_logits = decode_continuation(
                                        bad_weight_model,
                                        clone_inference_params(reference_cache),
                                        reference_prefill_last_logits,
                                        continuation_ids,
                                        device,
                                    )
                                    metrics = summarize_logits_diff(reference_logits, bad_weight_logits)
                                    loss, ppl = continuation_metrics(bad_weight_logits, continuation_ids)
                                    metrics["continuation_loss"] = loss
                                    metrics["continuation_ppl"] = ppl
                                else:
                                    bad_state_cache = clone_inference_params(reference_cache)
                                    bad_cache_warnings = []
                                    for layer_idx in group_target_layers:
                                        bad_cache_warnings.extend(
                                            permute_mamba2_cache_(bad_state_cache.key_value_memory_dict, layer_idx, perm_specs[layer_idx], debug=args.debug_layers)
                                        )
                                    bad_state_logits = decode_continuation(
                                        base_model,
                                        bad_state_cache,
                                        reference_prefill_last_logits,
                                        continuation_ids,
                                        device,
                                    )
                                    metrics = summarize_logits_diff(reference_logits, bad_state_logits)
                                    loss, ppl = continuation_metrics(bad_state_logits, continuation_ids)
                                    metrics["continuation_loss"] = loss
                                    metrics["continuation_ppl"] = ppl
                                    if bad_cache_warnings:
                                        metrics["warning_msg"] = " ; ".join(bad_cache_warnings)

                                status = classify_status(mode, metrics, good_metrics)
                                print(
                                    f"mode={mode} perm_type={perm_type} prompt_id={prompt_id} "
                                    f"max_abs_diff={metrics['max_abs_diff']:.8e} mean_abs_diff={metrics['mean_abs_diff']:.8e} "
                                    f"relative_max_error={metrics['relative_max_error']:.8e} top1_match_rate={metrics['top1_match_rate']:.8f} "
                                    f"continuation_loss={metrics['continuation_loss']:.8f} continuation_ppl={metrics['continuation_ppl']:.8f} status={status}"
                                )
                                row = {
                                    "perm_type": perm_type,
                                    "mode": mode,
                                    "prompt_id": prompt_id,
                                    "layers": args.layers,
                                    "layer_ids": args.layer_ids if args.layer_ids is not None else "",
                                    "heads": args.heads,
                                    "target_layers": str(group_target_layers),
                                    "target_heads": str(target_heads),
                                    "prefix_len": args.prefix_len,
                                    "continuation_len": args.continuation_len,
                                    "disable_mem_eff_path": args.disable_mem_eff_path,
                                    "max_abs_diff": f"{metrics['max_abs_diff']:.8e}",
                                    "mean_abs_diff": f"{metrics['mean_abs_diff']:.8e}",
                                    "relative_max_error": f"{metrics['relative_max_error']:.8e}",
                                    "top1_match_rate": f"{metrics['top1_match_rate']:.8f}",
                                    "continuation_loss": f"{metrics['continuation_loss']:.8f}",
                                    "continuation_ppl": f"{metrics['continuation_ppl']:.8f}",
                                    "status": status,
                                    "error_msg": metrics.get("warning_msg", ""),
                                }
                                append_row(row)
                                summary_by_mode[(perm_type, mode)].append(metrics | {"status": status})
                            except Exception as exc:
                                tb = traceback.format_exc()
                                print(f"[ERROR] mode={mode} perm_type={perm_type} prompt_id={prompt_id} exception={exc}")
                                print(tb)
                                row = make_error_row(
                                    args,
                                    perm_type,
                                    mode,
                                    prompt_id,
                                    group_target_layers,
                                    target_heads,
                                    f"{exc} | {tb.strip()}",
                                )
                                append_row(row)
                                if args.stop_on_error:
                                    raise
                    except Exception as exc:
                        tb = traceback.format_exc()
                        print(f"[ERROR] perm_type={perm_type} prompt_id={prompt_id} exception={exc}")
                        print(tb)
                        append_error_rows(perm_type, prompt_id, f"{exc} | {tb.strip()}")
                        if args.stop_on_error:
                            raise
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[FATAL] phase21 main loop exception={exc}")
        print(tb)
        fatal_row = make_error_row(
            args,
            "fatal",
            "error",
            -1,
            locals().get("target_layers", []),
            locals().get("target_heads", []),
            f"{exc} | {tb.strip()}",
        )
        fatal_row["layer_ids"] = args.layer_ids if args.layer_ids is not None else ""
        append_row(fatal_row)
        if args.stop_on_error:
            raise

    if not prompt_loop_executed:
        append_single_error("postflight", "error", -1, target_layers, target_heads, "prompt loop did not execute")

    if rows_written["count"] == 0:
        append_single_error("postflight", "error", -1, target_layers, target_heads, "No experiment results were written")

    print(f"Saved results to {output_path}")

    print("\nSummary")
    for (perm_type, mode), metrics_list in sorted(summary_by_mode.items()):
        if not metrics_list:
            print(f"perm_type={perm_type} mode={mode} no successful rows")
            continue
        avg_top1 = sum(item["top1_match_rate"] for item in metrics_list) / len(metrics_list)
        avg_rel = sum(item["relative_max_error"] for item in metrics_list) / len(metrics_list)
        avg_loss = sum(item["continuation_loss"] for item in metrics_list) / len(metrics_list)
        if mode == "good_weight_and_state_perm":
            pass_ratio = sum(item["status"] == "functional_pass" for item in metrics_list) / len(metrics_list)
            print(
                f"perm_type={perm_type} mode={mode} good pass ratio={pass_ratio:.2%} "
                f"avg_top1_match_rate={avg_top1:.8f} avg_relative_max_error={avg_rel:.8e} avg_continuation_loss={avg_loss:.8f}"
            )
        else:
            fail_ratio = sum(item["status"] == "expected_fail" for item in metrics_list) / len(metrics_list)
            print(
                f"perm_type={perm_type} mode={mode} bad fail ratio={fail_ratio:.2%} "
                f"avg_top1_match_rate={avg_top1:.8f} avg_relative_max_error={avg_rel:.8e} avg_continuation_loss={avg_loss:.8f}"
            )


if __name__ == "__main__":
    main()
