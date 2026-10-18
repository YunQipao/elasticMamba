import argparse
import copy
import csv
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from output_naming import build_output_path

from phase12_mamba2_head_permutation_equivalence import (
    MODEL_NAME,
    RMSNormGatedReference,
    collect_mamba2_mixers,
    configure_mamba2_for_equivalence,
    ensure_local_hf_cache,
)
from mamba2_permutation_utils import bad_permute_mamba2_inproj_only_, permute_mamba2_heads_


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


SEED = 0
BATCH_SIZE = 2
SEQ_LEN = 128
D_MODEL = 768


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def build_model(device: str, dtype: torch.dtype):
    try:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        model = MambaLMHeadModel.from_pretrained(MODEL_NAME, device=device, dtype=dtype)
        model.eval()
        return model, "mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel"
    except Exception as primary_exc:
        print(f"[WARN] MambaLMHeadModel loader failed: {primary_exc}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        return model, "transformers.AutoModelForCausalLM"


@torch.no_grad()
def summarize_output_diff(reference_output, compared_output):
    reference_cpu = reference_output.float().cpu()
    compared_cpu = compared_output.float().cpu()
    abs_diff = (reference_cpu - compared_cpu).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    reference_abs_max = reference_cpu.abs().max().item()
    relative_max_error = max_abs_diff / (reference_abs_max + 1e-12)
    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_max_error": relative_max_error,
        "output_abs_max": reference_abs_max,
    }


@torch.no_grad()
def print_metrics(tag: str, metrics: dict):
    print(tag)
    print(f"max_abs_diff       = {metrics['max_abs_diff']:.8e}")
    print(f"mean_abs_diff      = {metrics['mean_abs_diff']:.8e}")
    print(f"relative_max_error = {metrics['relative_max_error']:.8e}")


@torch.no_grad()
def maybe_set_mem_eff_path(mixer, enabled: bool):
    if hasattr(mixer, "use_mem_eff_path"):
        mixer.use_mem_eff_path = enabled


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    csv_path = build_output_path(
        args.output_csv,
        "phase12b_mamba2_block_equivalence",
        tags={"batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "d_model": D_MODEL},
    )
    ensure_local_hf_cache(csv_path.parent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading model: {MODEL_NAME}")
    model, loader_name = build_model(device=device, dtype=dtype)
    print(f"Model loader: {loader_name}")
    model = configure_mamba2_for_equivalence(model, disable_mem_eff_path=False)

    mixers = collect_mamba2_mixers(model)
    if not mixers:
        raise RuntimeError("No Mamba2 mixer layers found")

    layer_idx, base_mixer = mixers[0]
    print(f"Using layer {layer_idx} mixer with nheads={base_mixer.nheads} headdim={base_mixer.headdim}")

    hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device, dtype=dtype)
    perm = torch.randperm(base_mixer.nheads, device=base_mixer.in_proj.weight.device)

    rows = []
    for use_mem_eff_path in [True, False]:
        print(f"\nuse_mem_eff_path={use_mem_eff_path}")
        reference_mixer = copy.deepcopy(base_mixer).eval()
        good_mixer = copy.deepcopy(base_mixer).eval()
        bad_mixer = copy.deepcopy(base_mixer).eval()

        if not use_mem_eff_path:
            reference_mixer = configure_mamba2_for_equivalence(reference_mixer, disable_mem_eff_path=True)
            good_mixer = configure_mamba2_for_equivalence(good_mixer, disable_mem_eff_path=True)
            bad_mixer = configure_mamba2_for_equivalence(bad_mixer, disable_mem_eff_path=True)
        else:
            maybe_set_mem_eff_path(reference_mixer, True)
            maybe_set_mem_eff_path(good_mixer, True)
            maybe_set_mem_eff_path(bad_mixer, True)
            if getattr(reference_mixer, "norm", None) is not None and not isinstance(reference_mixer.norm, RMSNormGatedReference):
                pass

        reference_output = reference_mixer(hidden_states)

        permute_mamba2_heads_(good_mixer, perm)
        good_output = good_mixer(hidden_states)
        good_metrics = summarize_output_diff(reference_output, good_output)
        print_metrics("[GOOD] synchronized head permutation", good_metrics)
        rows.append(
            {
                "tag": "good",
                "use_mem_eff_path": use_mem_eff_path,
                "layer_idx": layer_idx,
                "batch_size": BATCH_SIZE,
                "seq_len": SEQ_LEN,
                "d_model": D_MODEL,
                "max_abs_diff": f"{good_metrics['max_abs_diff']:.8e}",
                "mean_abs_diff": f"{good_metrics['mean_abs_diff']:.8e}",
                "relative_max_error": f"{good_metrics['relative_max_error']:.8e}",
                "output_abs_max": f"{good_metrics['output_abs_max']:.8e}",
            }
        )

        bad_permute_mamba2_inproj_only_(bad_mixer, perm)
        bad_output = bad_mixer(hidden_states)
        bad_metrics = summarize_output_diff(reference_output, bad_output)
        print_metrics("[BAD] in_proj-only head permutation", bad_metrics)
        rows.append(
            {
                "tag": "bad",
                "use_mem_eff_path": use_mem_eff_path,
                "layer_idx": layer_idx,
                "batch_size": BATCH_SIZE,
                "seq_len": SEQ_LEN,
                "d_model": D_MODEL,
                "max_abs_diff": f"{bad_metrics['max_abs_diff']:.8e}",
                "mean_abs_diff": f"{bad_metrics['mean_abs_diff']:.8e}",
                "relative_max_error": f"{bad_metrics['relative_max_error']:.8e}",
                "output_abs_max": f"{bad_metrics['output_abs_max']:.8e}",
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "use_mem_eff_path",
                "layer_idx",
                "batch_size",
                "seq_len",
                "d_model",
                "max_abs_diff",
                "mean_abs_diff",
                "relative_max_error",
                "output_abs_max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
