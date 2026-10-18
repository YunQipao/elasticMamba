import argparse
import copy
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules import mamba_simple
from mamba_ssm.ops import selective_scan_interface

from phase1_stress_test_mamba1_permutation import permute_mamba1_expand_channels_

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


MODEL_NAME = "state-spaces/mamba-130m"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
PROMPT = "The quick brown fox jumps over the lazy dog. This sentence is used to verify Mamba permutation equivalence."
SEED = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="1", help="Number of Mamba layers to permute, or 'all'.")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--negative_control", action="store_true")
    parser.add_argument("--threshold_max", type=float, default=None)
    parser.add_argument("--threshold_mean", type=float, default=None)
    return parser.parse_args()


@torch.no_grad()
def prepare_runtime():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        mamba_simple.causal_conv1d_fn = None
        mamba_simple.selective_scan_fn = selective_scan_interface.selective_scan_ref
    return device


@torch.no_grad()
def collect_mamba_mixers(model: MambaLMHeadModel):
    mixers = []
    for layer_idx, layer in enumerate(model.backbone.layers):
        mixer = getattr(layer, "mixer", None)
        if isinstance(mixer, Mamba):
            mixers.append((layer_idx, mixer))
    return mixers


@torch.no_grad()
def bad_permute_only_inproj_(m: Mamba, perm: torch.Tensor):
    d_inner = m.d_inner
    perm = perm.to(m.in_proj.weight.device)
    row_perm = torch.cat([perm, perm + d_inner], dim=0)
    m.in_proj.weight.data = m.in_proj.weight.data[row_perm, :].contiguous()
    if m.in_proj.bias is not None:
        m.in_proj.bias.data = m.in_proj.bias.data[row_perm].contiguous()


@torch.no_grad()
def resolve_num_layers(layers_arg: str, total_layers: int):
    if layers_arg == "all":
        return total_layers
    num_layers = int(layers_arg)
    if num_layers < 1 or num_layers > total_layers:
        raise ValueError(f"--layers must be in [1, {total_layers}] or 'all', got: {layers_arg}")
    return num_layers


@torch.no_grad()
def compute_logits(model, input_ids):
    return model(input_ids).logits.float().cpu()


@torch.no_grad()
def summarize_logits_diff(reference_logits, compared_logits):
    abs_diff = (reference_logits - compared_logits).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    logits_abs_max = reference_logits.abs().max().item()
    relative_max_error = max_abs_diff / (logits_abs_max + 1e-12)
    reference_top1 = reference_logits.argmax(dim=-1)
    compared_top1 = compared_logits.argmax(dim=-1)
    top1_match_rate = (reference_top1 == compared_top1).float().mean().item()
    return {
        "logits_dtype": str(compared_logits.dtype),
        "logits_abs_max": logits_abs_max,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_max_error": relative_max_error,
        "top1_match_rate": top1_match_rate,
    }


def print_metrics(tag: str, metrics: dict):
    print(tag)
    print(f"logits dtype       = {metrics['logits_dtype']}")
    print(f"max_abs_diff       = {metrics['max_abs_diff']:.8e}")
    print(f"mean_abs_diff      = {metrics['mean_abs_diff']:.8e}")
    print(f"relative_max_error = {metrics['relative_max_error']:.8e}")
    print(f"top1_match_rate    = {metrics['top1_match_rate']:.8f}")


def save_results(payload: dict):
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase4_permutation_equivalence.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {output_path}")


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = prepare_runtime()
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    if args.threshold_max is None or args.threshold_mean is None:
        if args.dtype == "fp32":
            default_threshold_max = 5e-4
            default_threshold_mean = 5e-5
        else:
            default_threshold_max = 5e-3
            default_threshold_mean = 5e-4
        threshold_max = default_threshold_max if args.threshold_max is None else args.threshold_max
        threshold_mean = default_threshold_mean if args.threshold_mean is None else args.threshold_mean
    else:
        threshold_max = args.threshold_max
        threshold_mean = args.threshold_mean

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Loading model: {MODEL_NAME}")
    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, device=device, dtype=dtype)
    model.eval()

    mixers = collect_mamba_mixers(model)
    total_mamba_layers = len(mixers)
    num_layers = resolve_num_layers(args.layers, total_mamba_layers)

    print(f"Found {total_mamba_layers} Mamba mixer layers")
    print(f"Permuting first {num_layers} layer(s)")
    print(f"Runtime device={device} dtype={args.dtype}")

    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    original_logits = compute_logits(model, input_ids)
    results = {
        "model_name": MODEL_NAME,
        "tokenizer_name": TOKENIZER_NAME,
        "prompt": PROMPT,
        "device": device,
        "dtype": args.dtype,
        "layers": args.layers,
        "resolved_num_layers": num_layers,
        "negative_control": args.negative_control,
        "threshold_max_arg": args.threshold_max,
        "threshold_mean_arg": args.threshold_mean,
        "threshold_max": threshold_max,
        "threshold_mean": threshold_mean,
    }

    if args.negative_control:
        layer_idx, mixer = mixers[0]
        perm = torch.randperm(mixer.d_inner, device=mixer.in_proj.weight.device)

        good_model = copy.deepcopy(model).eval()
        good_mixer = collect_mamba_mixers(good_model)[layer_idx][1]
        permute_mamba1_expand_channels_(good_mixer, perm)
        good_logits = compute_logits(good_model, input_ids)

        bad_model = copy.deepcopy(model).eval()
        bad_mixer = collect_mamba_mixers(bad_model)[layer_idx][1]
        bad_permute_only_inproj_(bad_mixer, perm)
        bad_logits = compute_logits(bad_model, input_ids)
        print(f"Applied negative control on layer {layer_idx} with d_inner={mixer.d_inner}")
        good_metrics = summarize_logits_diff(original_logits, good_logits)
        bad_metrics = summarize_logits_diff(original_logits, bad_logits)
        print_metrics("[GOOD] synchronized permutation", good_metrics)
        print_metrics("[BAD] in_proj-only permutation", bad_metrics)
        results["good"] = good_metrics
        results["bad"] = bad_metrics
        results["negative_control_layer_idx"] = layer_idx
        results["negative_control_d_inner"] = mixer.d_inner

        assert bad_metrics["max_abs_diff"] > 100.0 * good_metrics["max_abs_diff"] or bad_metrics["max_abs_diff"] > 1e-2, (
            f"Negative control too weak: bad_max_abs_diff={bad_metrics['max_abs_diff']:.8e}"
        )
        save_results(results)
        print("[PASS] Negative control produced a clear mismatch.")
        return
    else:
        for layer_idx, mixer in mixers[:num_layers]:
            perm = torch.randperm(mixer.d_inner, device=mixer.in_proj.weight.device)
            permute_mamba1_expand_channels_(mixer, perm)
            print(f"Permuted layer {layer_idx} with d_inner={mixer.d_inner}")

    permuted_logits = compute_logits(model, input_ids)
    good_metrics = summarize_logits_diff(original_logits, permuted_logits)
    print_metrics("[GOOD] synchronized permutation", good_metrics)
    results["good"] = good_metrics

    assert good_metrics["max_abs_diff"] < threshold_max and good_metrics["mean_abs_diff"] < threshold_mean, (
        f"Permutation equivalence check failed for dtype={args.dtype}: "
        f"max_abs_diff={good_metrics['max_abs_diff']:.8e} >= {threshold_max:.8e} or "
        f"mean_abs_diff={good_metrics['mean_abs_diff']:.8e} >= {threshold_mean:.8e}"
    )

    save_results(results)
    print("[PASS] Pretrained Mamba LM permutation equivalence verified.")


if __name__ == "__main__":
    main()
