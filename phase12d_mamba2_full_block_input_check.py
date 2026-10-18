import argparse
import copy
import csv
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer
from output_naming import build_output_path

from mamba2_permutation_utils import bad_permute_mamba2_inproj_only_, permute_mamba2_heads_
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
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


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


@torch.no_grad()
def capture_layer0_input(model, input_ids):
    captured = {}

    def pre_hook(_module, inputs):
        hidden = inputs[0]
        captured["hidden"] = hidden.detach().clone()

    handle = model.backbone.layers[0].mixer.register_forward_pre_hook(pre_hook)
    try:
        _ = model(input_ids)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("Failed to capture layer 0 mixer input")
    return captured["hidden"]


@torch.no_grad()
def print_metrics(tag: str, metrics: dict):
    print(tag)
    print(f"max_abs_diff       = {metrics['max_abs_diff']:.8e}")
    print(f"mean_abs_diff      = {metrics['mean_abs_diff']:.8e}")
    print(f"relative_max_error = {metrics['relative_max_error']:.8e}")


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    csv_path = build_output_path(args.output_csv, "phase12d_mamba2_full_block_input_check")
    ensure_local_hf_cache(csv_path.parent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    base_model, loader_name = build_model(device=device, dtype=dtype)
    print(f"Model loader: {loader_name}")

    for disable_mem_eff_path in [False, True]:
        print(f"\ndisable_mem_eff_path={disable_mem_eff_path}")
        model = configure_mamba2_for_equivalence(copy.deepcopy(base_model).eval(), disable_mem_eff_path=disable_mem_eff_path)
        encoded = tokenizer(PROMPT, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        layer0_input = capture_layer0_input(model, input_ids)
        print(f"Captured layer 0 mixer input shape={tuple(layer0_input.shape)}")

        reference_mixer = copy.deepcopy(collect_mamba2_mixers(model)[0][1]).eval()
        good_mixer = copy.deepcopy(reference_mixer).eval()
        bad_mixer = copy.deepcopy(reference_mixer).eval()

        perm = torch.randperm(reference_mixer.nheads, device=reference_mixer.in_proj.weight.device)
        reference_output = reference_mixer(layer0_input)

        permute_mamba2_heads_(good_mixer, perm)
        good_output = good_mixer(layer0_input)
        good_metrics = summarize_hidden_diff(reference_output, good_output)
        print_metrics("[GOOD] full-model layer0 input -> synchronized mixer", good_metrics)

        bad_permute_mamba2_inproj_only_(bad_mixer, perm)
        bad_output = bad_mixer(layer0_input)
        bad_metrics = summarize_hidden_diff(reference_output, bad_output)
        print_metrics("[BAD] full-model layer0 input -> in_proj-only mixer", bad_metrics)

        rows = [
            {
                "tag": "good",
                "disable_mem_eff_path": disable_mem_eff_path,
                "max_abs_diff": f"{good_metrics['max_abs_diff']:.8e}",
                "mean_abs_diff": f"{good_metrics['mean_abs_diff']:.8e}",
                "relative_max_error": f"{good_metrics['relative_max_error']:.8e}",
                "abs_max": f"{good_metrics['hidden_abs_max']:.8e}",
            },
            {
                "tag": "bad",
                "disable_mem_eff_path": disable_mem_eff_path,
                "max_abs_diff": f"{bad_metrics['max_abs_diff']:.8e}",
                "mean_abs_diff": f"{bad_metrics['mean_abs_diff']:.8e}",
                "relative_max_error": f"{bad_metrics['relative_max_error']:.8e}",
                "abs_max": f"{bad_metrics['hidden_abs_max']:.8e}",
            },
        ]

        mode = "a" if csv_path.exists() else "w"
        with csv_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["tag", "disable_mem_eff_path", "max_abs_diff", "mean_abs_diff", "relative_max_error", "abs_max"],
            )
            if mode == "w":
                writer.writeheader()
            writer.writerows(rows)

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
