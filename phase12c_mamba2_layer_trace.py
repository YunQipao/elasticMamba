import argparse
import copy
import csv
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from output_naming import build_output_path

from mamba2_permutation_utils import permute_mamba2_heads_
from phase12_mamba2_head_permutation_equivalence import (
    MODEL_NAME,
    PROMPT,
    TOKENIZER_NAME,
    build_model,
    collect_mamba2_mixers,
    configure_mamba2_for_equivalence,
    ensure_local_hf_cache,
    summarize_logits_diff,
)


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


SEED = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="1")
    parser.add_argument("--disable_mem_eff_path", action="store_true")
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def resolve_num_layers(layers_arg: str, total_layers: int):
    if layers_arg == "all":
        return total_layers
    value = int(layers_arg)
    if value < 1 or value > total_layers:
        raise ValueError(f"--layers must be in [1, {total_layers}] or 'all', got {layers_arg}")
    return value


@torch.no_grad()
def summarize_hidden_diff(reference_hidden, compared_hidden):
    reference_cpu = reference_hidden.float().cpu()
    compared_cpu = compared_hidden.float().cpu()
    abs_diff = (reference_cpu - compared_cpu).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    reference_abs_max = reference_cpu.abs().max().item()
    relative_max_error = max_abs_diff / (reference_abs_max + 1e-12)
    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_max_error": relative_max_error,
        "hidden_abs_max": reference_abs_max,
    }


def register_layer_hooks(model, storage):
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            storage[layer_idx] = hidden.detach().float().cpu()
        return hook

    for layer_idx, layer in enumerate(model.backbone.layers):
        handles.append(layer.register_forward_hook(make_hook(layer_idx)))
    return handles


@torch.no_grad()
def compute_logits_with_trace(model, input_ids):
    traced = {}
    handles = register_layer_hooks(model, traced)
    try:
        logits = model(input_ids).logits.float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return logits, traced


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    csv_path = build_output_path(
        args.output_csv,
        "phase12c_mamba2_layer_trace",
        tags={"layers": args.layers, "disable_mem_eff": args.disable_mem_eff_path},
    )
    ensure_local_hf_cache(csv_path.parent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    base_model, loader_name = build_model(device=device, dtype=dtype)
    print(f"Model loader: {loader_name}")

    original_model = configure_mamba2_for_equivalence(
        copy.deepcopy(base_model).eval(), disable_mem_eff_path=args.disable_mem_eff_path
    )
    permuted_model = configure_mamba2_for_equivalence(
        copy.deepcopy(base_model).eval(), disable_mem_eff_path=args.disable_mem_eff_path
    )

    original_mixers = collect_mamba2_mixers(original_model)
    permuted_mixers = collect_mamba2_mixers(permuted_model)
    total_layers = len(permuted_mixers)
    if total_layers == 0:
        raise RuntimeError("No Mamba2 mixer layers found")
    num_layers = resolve_num_layers(args.layers, total_layers)
    print(f"Tracing first {num_layers} permuted layer(s)")

    for layer_idx, mixer in permuted_mixers[:num_layers]:
        perm = torch.randperm(int(getattr(mixer, "nheads")), device=mixer.in_proj.weight.device)
        permute_mamba2_heads_(mixer, perm)
        print(f"Permuted layer {layer_idx} with nheads={mixer.nheads} headdim={mixer.headdim}")

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    original_logits, original_trace = compute_logits_with_trace(original_model, input_ids)
    permuted_logits, permuted_trace = compute_logits_with_trace(permuted_model, input_ids)

    rows = []
    for layer_idx in range(len(original_model.backbone.layers)):
        if layer_idx not in original_trace or layer_idx not in permuted_trace:
            continue
        metrics = summarize_hidden_diff(original_trace[layer_idx], permuted_trace[layer_idx])
        print(
            f"layer={layer_idx} max_abs_diff={metrics['max_abs_diff']:.8e} "
            f"mean_abs_diff={metrics['mean_abs_diff']:.8e} "
            f"relative_max_error={metrics['relative_max_error']:.8e}"
        )
        rows.append(
            {
                "kind": "layer_hidden",
                "layer_idx": layer_idx,
                "layers_arg": args.layers,
                "num_permuted_layers": num_layers,
                "disable_mem_eff_path": args.disable_mem_eff_path,
                "max_abs_diff": f"{metrics['max_abs_diff']:.8e}",
                "mean_abs_diff": f"{metrics['mean_abs_diff']:.8e}",
                "relative_max_error": f"{metrics['relative_max_error']:.8e}",
                "abs_max": f"{metrics['hidden_abs_max']:.8e}",
            }
        )

    logits_metrics = summarize_logits_diff(original_logits, permuted_logits)
    print(
        f"logits max_abs_diff={logits_metrics['max_abs_diff']:.8e} "
        f"mean_abs_diff={logits_metrics['mean_abs_diff']:.8e} "
        f"relative_max_error={logits_metrics['relative_max_error']:.8e} "
        f"top1_match_rate={logits_metrics['top1_match_rate']:.8f}"
    )
    rows.append(
        {
            "kind": "logits",
            "layer_idx": -1,
            "layers_arg": args.layers,
            "num_permuted_layers": num_layers,
            "disable_mem_eff_path": args.disable_mem_eff_path,
            "max_abs_diff": f"{logits_metrics['max_abs_diff']:.8e}",
            "mean_abs_diff": f"{logits_metrics['mean_abs_diff']:.8e}",
            "relative_max_error": f"{logits_metrics['relative_max_error']:.8e}",
            "abs_max": f"{logits_metrics['logits_abs_max']:.8e}",
        }
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kind",
                "layer_idx",
                "layers_arg",
                "num_permuted_layers",
                "disable_mem_eff_path",
                "max_abs_diff",
                "mean_abs_diff",
                "relative_max_error",
                "abs_max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
