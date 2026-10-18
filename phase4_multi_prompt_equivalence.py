import argparse
import copy
import csv
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules import mamba_simple
from mamba_ssm.ops import selective_scan_interface
from output_naming import build_output_path

from phase1_stress_test_mamba1_permutation import permute_mamba1_expand_channels_


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


MODEL_NAME = "state-spaces/mamba-130m"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
SEED = 0
PROMPTS = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Mamba models are efficient sequence models for long-context reasoning.",
    "In a quiet village, the river bends around old stones and willow trees, carrying reflections of the sky while children race along the bank and laughter fades into the evening air.",
    "```python\ndef add(a, b):\n    return a + b\n```\nThis snippet defines a tiny helper function.",
    "Let f(x) = x^2 + 3x + 1. By completing the square, we can analyze its minima and curvature.",
    "To prepare the mixture, combine flour, water, yeast, and salt, then let the dough rest overnight before baking in a hot oven until the crust becomes golden and crisp.",
    "A proof by contradiction assumes the negation of the claim and derives an inconsistency, often revealing the structure hidden beneath the problem statement.",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[32, 128, 512, 1024])
    parser.add_argument("--output_csv", type=str, default=None)
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
def compute_logits(model, input_ids):
    return model(input_ids).logits.float().cpu()


@torch.no_grad()
def summarize_logits_diff(reference_logits, compared_logits):
    abs_diff = (reference_logits - compared_logits).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    logits_abs_max = reference_logits.abs().max().item()
    relative_max_error = max_abs_diff / (logits_abs_max + 1e-12)
    top1_match_rate = (reference_logits.argmax(dim=-1) == compared_logits.argmax(dim=-1)).float().mean().item()
    return {
        "logits_dtype": str(compared_logits.dtype),
        "logits_abs_max": logits_abs_max,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_max_error": relative_max_error,
        "top1_match_rate": top1_match_rate,
    }


def is_pass(metrics, threshold_max, threshold_mean, threshold_relative_max=1e-5):
    return (
        metrics["max_abs_diff"] < threshold_max
        and metrics["mean_abs_diff"] < threshold_mean
        and metrics["relative_max_error"] < threshold_relative_max
        and metrics["top1_match_rate"] >= 1.0 - 1e-8
    )


@torch.no_grad()
def run_with_perm(model, mixers, input_ids, num_layers):
    permuted_model = copy.deepcopy(model).eval()
    permuted_mixers = collect_mamba_mixers(permuted_model)
    for layer_idx, mixer in permuted_mixers[:num_layers]:
        perm = torch.randperm(mixer.d_inner, device=mixer.in_proj.weight.device)
        permute_mamba1_expand_channels_(mixer, perm)
    return compute_logits(permuted_model, input_ids)


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = prepare_runtime()
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    if args.dtype == "fp32":
        threshold_max = 8e-4
        threshold_mean = 1e-4
    else:
        threshold_max = 5e-3
        threshold_mean = 5e-4

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, device=device, dtype=dtype)
    model.eval()

    mixers = collect_mamba_mixers(model)
    total_layers = len(mixers)
    if total_layers != 24:
        print(f"[WARN] Expected 24 Mamba layers, found {total_layers}")

    csv_path = build_output_path(
        args.output_csv,
        "phase4_multi_prompt_equivalence",
        tags={"dtype": args.dtype, "seq_lens": "-".join(map(str, args.seq_lens))},
    )

    rows = []
    total_cases = 0
    passed_cases = 0

    for prompt_idx, prompt in enumerate(PROMPTS):
        for seq_len in args.seq_lens:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len)
            input_ids = inputs["input_ids"].to(device)
            original_logits = compute_logits(model, input_ids)

            for layers_label, num_layers in [("1", 1), ("all", total_layers)]:
                permuted_logits = run_with_perm(model, mixers, input_ids, num_layers)
                metrics = summarize_logits_diff(original_logits, permuted_logits)
                ok = is_pass(metrics, threshold_max, threshold_mean)
                total_cases += 1
                passed_cases += int(ok)

                print(
                    f"prompt={prompt_idx} seq_len={seq_len} layers={layers_label} "
                    f"max_abs_diff={metrics['max_abs_diff']:.8e} "
                    f"mean_abs_diff={metrics['mean_abs_diff']:.8e} "
                    f"relative_max_error={metrics['relative_max_error']:.8e} "
                    f"top1_match_rate={metrics['top1_match_rate']:.8f} status={'PASS' if ok else 'FAIL'}"
                )

                rows.append(
                    {
                        "prompt_idx": prompt_idx,
                        "prompt": prompt,
                        "seq_len": seq_len,
                        "layers": layers_label,
                        "dtype": args.dtype,
                        "logits_dtype": metrics["logits_dtype"],
                        "logits_abs_max": f"{metrics['logits_abs_max']:.8e}",
                        "max_abs_diff": f"{metrics['max_abs_diff']:.8e}",
                        "mean_abs_diff": f"{metrics['mean_abs_diff']:.8e}",
                        "relative_max_error": f"{metrics['relative_max_error']:.8e}",
                        "top1_match_rate": f"{metrics['top1_match_rate']:.8f}",
                        "threshold_max": f"{threshold_max:.8e}",
                        "threshold_mean": f"{threshold_mean:.8e}",
                        "pass": str(ok),
                    }
                )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_idx",
                "prompt",
                "seq_len",
                "layers",
                "dtype",
                "logits_dtype",
                "logits_abs_max",
                "max_abs_diff",
                "mean_abs_diff",
                "relative_max_error",
                "top1_match_rate",
                "threshold_max",
                "threshold_mean",
                "pass",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {csv_path}")
    print(f"Pass rate: {passed_cases}/{total_cases} = {passed_cases / total_cases:.2%}")
    assert passed_cases == total_cases, "Some prompt/length configurations failed the permutation checks."
    print("[PASS] Multi-prompt permutation equivalence verified.")


if __name__ == "__main__":
    main()
