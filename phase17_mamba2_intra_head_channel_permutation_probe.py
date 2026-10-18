import argparse
import copy
import csv
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer
from output_naming import build_output_path

from mamba2_permutation_utils import get_mamba2_layout
from phase12_mamba2_head_permutation_equivalence import (
    MODEL_NAME,
    PROMPT,
    TOKENIZER_NAME,
    build_model,
    collect_mamba2_mixers,
    configure_mamba2_for_equivalence,
    ensure_local_hf_cache,
)


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


STRICT_MAX = 1e-4
STRICT_MEAN = 1e-5
STRICT_REL = 1e-6
FUNCTIONAL_MAX = 1e-3
FUNCTIONAL_MEAN = 1e-4
FUNCTIONAL_REL = 1e-5
NEGATIVE_RATIO = 1000.0
NEGATIVE_MAX_FLOOR = 1e-2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--head_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--disable_mem_eff_path", action="store_true")
    parser.add_argument("--test_all_heads", action="store_true")
    parser.add_argument("--test_layers", nargs="+", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def make_intra_head_channel_perm(mixer, head_id: int, seed: int):
    headdim = int(getattr(mixer, "headdim"))
    generator = torch.Generator(device=mixer.in_proj.weight.device)
    generator.manual_seed(int(seed) + int(head_id) * 1009)
    return torch.randperm(headdim, generator=generator, device=mixer.in_proj.weight.device)


@torch.no_grad()
def _permute_linear_rows_slice_(linear, start: int, end: int, index: torch.Tensor):
    linear.weight.data[start:end, :] = linear.weight.data[start:end, :][index, :].contiguous()
    if linear.bias is not None:
        linear.bias.data[start:end] = linear.bias.data[start:end][index].contiguous()


@torch.no_grad()
def _permute_conv1d_channels_slice_(conv1d, start: int, end: int, index: torch.Tensor):
    conv1d.weight.data[start:end, :, :] = conv1d.weight.data[start:end, :, :][index, :, :].contiguous()
    if conv1d.bias is not None:
        conv1d.bias.data[start:end] = conv1d.bias.data[start:end][index].contiguous()


@torch.no_grad()
def _permute_norm_slice_(norm_module, start: int, end: int, index: torch.Tensor):
    if norm_module is None:
        return
    if hasattr(norm_module, "weight") and isinstance(norm_module.weight, torch.Tensor):
        norm_module.weight.data[start:end] = norm_module.weight.data[start:end][index].contiguous()
    if hasattr(norm_module, "bias") and isinstance(norm_module.bias, torch.Tensor):
        norm_module.bias.data[start:end] = norm_module.bias.data[start:end][index].contiguous()


@torch.no_grad()
def _permute_out_proj_input_slice_(out_proj, start: int, end: int, index: torch.Tensor):
    out_proj.weight.data[:, start:end] = out_proj.weight.data[:, start:end][:, index].contiguous()


@torch.no_grad()
def permute_mamba2_intra_head_channels_(mixer, head_id: int, perm: torch.Tensor):
    layout = get_mamba2_layout(mixer)
    nheads = layout["nheads"]
    headdim = layout["headdim"]
    if head_id < 0 or head_id >= nheads:
        raise ValueError(f"head_id must be in [0, {nheads - 1}], got {head_id}")
    if perm.numel() != headdim:
        raise ValueError(f"Expected intra-head perm size {headdim}, got {perm.numel()}")

    c0 = head_id * headdim
    c1 = (head_id + 1) * headdim

    z_start, _ = layout["in_proj"]["z"]
    x_start, _ = layout["in_proj"]["x"]
    conv_x_start, _ = layout["conv1d"]["x"]

    _permute_linear_rows_slice_(mixer.in_proj, z_start + c0, z_start + c1, perm)
    _permute_linear_rows_slice_(mixer.in_proj, x_start + c0, x_start + c1, perm)
    _permute_conv1d_channels_slice_(mixer.conv1d, conv_x_start + c0, conv_x_start + c1, perm)
    _permute_norm_slice_(getattr(mixer, "norm", None), c0, c1, perm)
    _permute_out_proj_input_slice_(mixer.out_proj, c0, c1, perm)


@torch.no_grad()
def bad_permute_mamba2_intra_head_inproj_only_(mixer, head_id: int, perm: torch.Tensor):
    layout = get_mamba2_layout(mixer)
    headdim = layout["headdim"]
    nheads = layout["nheads"]
    if head_id < 0 or head_id >= nheads:
        raise ValueError(f"head_id must be in [0, {nheads - 1}], got {head_id}")
    if perm.numel() != headdim:
        raise ValueError(f"Expected intra-head perm size {headdim}, got {perm.numel()}")

    c0 = head_id * headdim
    c1 = (head_id + 1) * headdim
    z_start, _ = layout["in_proj"]["z"]
    x_start, _ = layout["in_proj"]["x"]

    _permute_linear_rows_slice_(mixer.in_proj, z_start + c0, z_start + c1, perm)
    _permute_linear_rows_slice_(mixer.in_proj, x_start + c0, x_start + c1, perm)


@torch.no_grad()
def capture_mixer_input(model, input_ids, layer_idx: int):
    captured = {}

    def pre_hook(_module, inputs):
        captured["hidden"] = inputs[0].detach().clone()

    handle = model.backbone.layers[layer_idx].mixer.register_forward_pre_hook(pre_hook)
    try:
        _ = model(input_ids)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError(f"Failed to capture mixer input for layer {layer_idx}")
    return captured["hidden"]


@torch.no_grad()
def summarize_diff(reference, compared):
    reference_cpu = reference.float().cpu()
    compared_cpu = compared.float().cpu()
    abs_diff = (reference_cpu - compared_cpu).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    output_abs_max = reference_cpu.abs().max().item()
    relative_max_error = max_abs_diff / (output_abs_max + 1e-12)
    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_max_error": relative_max_error,
        "output_abs_max": output_abs_max,
    }


@torch.no_grad()
def classify_good(metrics: dict):
    if (
        metrics["max_abs_diff"] < STRICT_MAX
        and metrics["mean_abs_diff"] < STRICT_MEAN
        and metrics["relative_max_error"] < STRICT_REL
    ):
        return "strict_pass"
    if (
        metrics["max_abs_diff"] < FUNCTIONAL_MAX
        and metrics["mean_abs_diff"] < FUNCTIONAL_MEAN
        and metrics["relative_max_error"] < FUNCTIONAL_REL
    ):
        return "functional_pass"
    return "fail"


@torch.no_grad()
def classify_bad(good_metrics: dict, bad_metrics: dict):
    good_max = max(good_metrics["max_abs_diff"], 1e-12)
    if bad_metrics["max_abs_diff"] > NEGATIVE_MAX_FLOOR or bad_metrics["max_abs_diff"] >= NEGATIVE_RATIO * good_max:
        return "expected_fail"
    return "weak_fail"


@torch.no_grad()
def set_all_mixer_mem_eff_path(model, enabled: bool):
    for _, mixer in collect_mamba2_mixers(model):
        if hasattr(mixer, "use_mem_eff_path"):
            mixer.use_mem_eff_path = enabled


@torch.no_grad()
def print_case(tag: str, input_type: str, layer_idx: int, head_id: int, use_mem_eff_path: bool, metrics: dict, status: str):
    print(
        f"[{tag.upper()}] input_type={input_type} layer={layer_idx} head={head_id} "
        f"use_mem_eff_path={use_mem_eff_path} "
        f"max_abs_diff={metrics['max_abs_diff']:.8e} "
        f"mean_abs_diff={metrics['mean_abs_diff']:.8e} "
        f"relative_max_error={metrics['relative_max_error']:.8e} "
        f"output_abs_max={metrics['output_abs_max']:.8e} status={status}"
    )


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_path = build_output_path(
        args.output_csv,
        "phase17_mamba2_intra_head_channel_permutation_probe",
        tags={
            "layers": "-".join(map(str, args.test_layers)) if args.test_layers is not None else args.layer_idx,
            "heads": "all" if args.test_all_heads else args.head_id,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "disable_mem_eff": args.disable_mem_eff_path,
        },
    )
    ensure_local_hf_cache(output_path.parent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    model, loader_name = build_model(device=device, dtype=dtype)
    print(f"Model loader: {loader_name}")
    model = configure_mamba2_for_equivalence(model, disable_mem_eff_path=False)

    mixer_pairs = collect_mamba2_mixers(model)
    if not mixer_pairs:
        raise RuntimeError("No Mamba2 mixer layers found")

    num_layers = len(mixer_pairs)
    target_layers = args.test_layers if args.test_layers is not None else [args.layer_idx]
    invalid_layers = [layer_idx for layer_idx in target_layers if layer_idx < 0 or layer_idx >= num_layers]
    if invalid_layers:
        raise ValueError(f"Invalid layers: {invalid_layers}; available range is [0, {num_layers - 1}]")

    first_mixer = mixer_pairs[target_layers[0]][1]
    total_heads = int(getattr(first_mixer, "nheads"))
    target_heads = list(range(total_heads)) if args.test_all_heads else [args.head_id]
    invalid_heads = [head_id for head_id in target_heads if head_id < 0 or head_id >= total_heads]
    if invalid_heads:
        raise ValueError(f"Invalid heads: {invalid_heads}; available range is [0, {total_heads - 1}]")

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    use_mem_eff_path_options = [False] if args.disable_mem_eff_path else [True, False]
    rows = []
    summary = {}

    for use_mem_eff_path in use_mem_eff_path_options:
        print(f"\n=== use_mem_eff_path={use_mem_eff_path} ===")
        set_all_mixer_mem_eff_path(model, use_mem_eff_path)

        captured_inputs = {}
        for layer_idx in target_layers:
            captured_inputs[layer_idx] = capture_mixer_input(model, input_ids, layer_idx)
            print(f"Captured layer {layer_idx} mixer input shape={tuple(captured_inputs[layer_idx].shape)}")

        for layer_idx in target_layers:
            source_mixer = mixer_pairs[layer_idx][1]
            d_model = int(getattr(source_mixer, "d_model", source_mixer.in_proj.weight.shape[1]))
            random_hidden = torch.randn(args.batch_size, args.seq_len, d_model, device=device, dtype=dtype)
            inputs_by_type = {
                "random_hidden": random_hidden,
                "full_model_captured_input": captured_inputs[layer_idx],
            }

            for head_id in target_heads:
                perm = make_intra_head_channel_perm(source_mixer, head_id=head_id, seed=args.seed + layer_idx * 10007)
                summary.setdefault((layer_idx, head_id), {"good": [], "bad": []})

                for input_type, hidden_states in inputs_by_type.items():
                    reference_mixer = copy.deepcopy(source_mixer).eval()
                    good_mixer = copy.deepcopy(source_mixer).eval()
                    bad_mixer = copy.deepcopy(source_mixer).eval()

                    reference_output = reference_mixer(hidden_states)

                    permute_mamba2_intra_head_channels_(good_mixer, head_id, perm)
                    good_output = good_mixer(hidden_states)
                    good_metrics = summarize_diff(reference_output, good_output)
                    good_status = classify_good(good_metrics)
                    print_case("good", input_type, layer_idx, head_id, use_mem_eff_path, good_metrics, good_status)

                    bad_permute_mamba2_intra_head_inproj_only_(bad_mixer, head_id, perm)
                    bad_output = bad_mixer(hidden_states)
                    bad_metrics = summarize_diff(reference_output, bad_output)
                    bad_status = classify_bad(good_metrics, bad_metrics)
                    print_case("bad", input_type, layer_idx, head_id, use_mem_eff_path, bad_metrics, bad_status)

                    rows.append(
                        {
                            "input_type": input_type,
                            "layer_idx": layer_idx,
                            "head_id": head_id,
                            "use_mem_eff_path": use_mem_eff_path,
                            "tag": "good",
                            "max_abs_diff": f"{good_metrics['max_abs_diff']:.8e}",
                            "mean_abs_diff": f"{good_metrics['mean_abs_diff']:.8e}",
                            "relative_max_error": f"{good_metrics['relative_max_error']:.8e}",
                            "output_abs_max": f"{good_metrics['output_abs_max']:.8e}",
                            "status": good_status,
                        }
                    )
                    rows.append(
                        {
                            "input_type": input_type,
                            "layer_idx": layer_idx,
                            "head_id": head_id,
                            "use_mem_eff_path": use_mem_eff_path,
                            "tag": "bad",
                            "max_abs_diff": f"{bad_metrics['max_abs_diff']:.8e}",
                            "mean_abs_diff": f"{bad_metrics['mean_abs_diff']:.8e}",
                            "relative_max_error": f"{bad_metrics['relative_max_error']:.8e}",
                            "output_abs_max": f"{bad_metrics['output_abs_max']:.8e}",
                            "status": bad_status,
                        }
                    )

                    summary[(layer_idx, head_id)]["good"].append(good_status)
                    summary[(layer_idx, head_id)]["bad"].append(bad_status)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_type",
                "layer_idx",
                "head_id",
                "use_mem_eff_path",
                "tag",
                "max_abs_diff",
                "mean_abs_diff",
                "relative_max_error",
                "output_abs_max",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {output_path}")
    print("\nSummary by head")
    all_good_strict = True
    any_good_strict = False
    all_bad_expected = True
    for (layer_idx, head_id), result in sorted(summary.items()):
        good_all_strict = all(status == "strict_pass" for status in result["good"])
        good_any_strict = any(status == "strict_pass" for status in result["good"])
        bad_all_expected = all(status == "expected_fail" for status in result["bad"])
        print(
            f"layer={layer_idx} head={head_id} "
            f"good={','.join(result['good'])} "
            f"bad={','.join(result['bad'])} "
            f"good_all_strict={good_all_strict} bad_all_expected={bad_all_expected}"
        )
        all_good_strict = all_good_strict and good_all_strict
        any_good_strict = any_good_strict or good_any_strict
        all_bad_expected = all_bad_expected and bad_all_expected

    if all_good_strict and all_bad_expected:
        print("[PASS] Mamba2 intra-head channel permutation consistency verified.")
    elif any_good_strict:
        print("[PARTIAL] Intra-head channel consistency holds for some heads.")
    else:
        print("[FAIL] Intra-head channel permutation consistency not verified.")


if __name__ == "__main__":
    main()
