import argparse
import copy
import csv
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from output_naming import build_output_path

from mamba2_permutation_utils import bad_permute_mamba2_inproj_only_, permute_mamba2_heads_


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


MODEL_NAME = "state-spaces/mamba2-130m"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is used to verify Mamba2 head permutation equivalence."
)
SEED = 0
THRESHOLD_MAX = 1e-3
THRESHOLD_MEAN = 1e-4
THRESHOLD_RELATIVE = 1e-5
THRESHOLD_TOP1 = 1.0


class RMSNormGatedReference(nn.Module):
    def __init__(self, norm_module, rms_norm_ref_fn):
        super().__init__()
        self.weight = norm_module.weight
        if getattr(norm_module, "bias", None) is None:
            self.register_parameter("bias", None)
        else:
            self.bias = norm_module.bias
        self.eps = norm_module.eps
        self.group_size = norm_module.group_size
        self.norm_before_gate = norm_module.norm_before_gate
        self._rms_norm_ref_fn = rms_norm_ref_fn

    def forward(self, x, z=None):
        return self._rms_norm_ref_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            upcast=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="1", help="Number of Mamba2 layers to permute, or 'all'.")
    parser.add_argument("--negative_control", action="store_true")
    parser.add_argument("--disable_mem_eff_path", action="store_true")
    parser.add_argument("--determinism_check", action="store_true")
    parser.add_argument(
        "--ablate",
        type=str,
        default=None,
        choices=["no_norm", "no_dt_bias", "no_D", "no_A_log", "no_conv", "no_out_proj", "permute_BC"],
    )
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def ensure_local_hf_cache(output_dir: Path):
    cache_root = output_dir / "hf_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))


@torch.no_grad()
def build_model(device: str, dtype: torch.dtype, local_files_only: bool = False):
    try:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        if local_files_only:
            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        model = MambaLMHeadModel.from_pretrained(MODEL_NAME, device=device, dtype=dtype)
        model.eval()
        return model, "mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel"
    except Exception as primary_exc:
        print(f"[WARN] MambaLMHeadModel loader failed: {primary_exc}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=dtype,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            model.to(device)
            model.eval()
            return model, "transformers.AutoModelForCausalLM"
        except Exception as fallback_exc:
            raise RuntimeError(
                "Failed to load model with both mamba_ssm and transformers loaders. "
                f"Primary loader error: {primary_exc}. Fallback loader error: {fallback_exc}"
            ) from fallback_exc


@torch.no_grad()
def collect_mamba2_mixers(model):
    mixers = []
    if hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
        for layer_idx, layer in enumerate(model.backbone.layers):
            mixer = getattr(layer, "mixer", None)
            if mixer is not None and mixer.__class__.__name__ == "Mamba2":
                mixers.append((layer_idx, mixer))
    if not mixers:
        for name, module in model.named_modules():
            if name.endswith(".mixer") and module.__class__.__name__ == "Mamba2":
                mixers.append((len(mixers), module))
    return mixers


def configure_mamba2_for_equivalence(model, disable_mem_eff_path: bool = False):
    from mamba_ssm.modules import mamba2 as mamba2_module
    from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref

    mamba2_module.causal_conv1d_fn = None
    for layer_idx, mixer in collect_mamba2_mixers(model):
        if disable_mem_eff_path:
            mixer.use_mem_eff_path = False
        if getattr(mixer, "norm", None) is not None and not isinstance(mixer.norm, RMSNormGatedReference):
            mixer.norm = RMSNormGatedReference(mixer.norm, rms_norm_ref)
        print(f"layer {layer_idx} use_mem_eff_path={getattr(mixer, 'use_mem_eff_path', None)}")
    return model


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


@torch.no_grad()
def print_metrics(tag: str, metrics: dict):
    print(tag)
    print(f"logits dtype       = {metrics['logits_dtype']}")
    print(f"max_abs_diff       = {metrics['max_abs_diff']:.8e}")
    print(f"mean_abs_diff      = {metrics['mean_abs_diff']:.8e}")
    print(f"relative_max_error = {metrics['relative_max_error']:.8e}")
    print(f"top1_match_rate    = {metrics['top1_match_rate']:.8f}")


@torch.no_grad()
def save_results_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "model_name",
                "tokenizer_name",
                "layers_arg",
                "num_layers",
                "negative_control",
                "determinism_check",
                "ablate",
                "max_abs_diff",
                "mean_abs_diff",
                "relative_max_error",
                "top1_match_rate",
                "logits_abs_max",
                "threshold_max",
                "threshold_mean",
                "threshold_relative",
                "threshold_top1",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {output_path}")


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    output_path = build_output_path(
        args.output_csv,
        "phase12_mamba2_head_permutation_equivalence",
        tags={
            "layers": args.layers,
            "neg": args.negative_control,
            "disable_mem_eff": args.disable_mem_eff_path,
            "determinism": args.determinism_check,
            "ablate": args.ablate or "none",
        },
    )
    ensure_local_hf_cache(output_path.parent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    model, loader_name = build_model(device=device, dtype=dtype)
    model = configure_mamba2_for_equivalence(model, disable_mem_eff_path=args.disable_mem_eff_path)
    print(f"Model loader: {loader_name}")

    mixers = collect_mamba2_mixers(model)
    total_layers = len(mixers)
    print(f"Found {total_layers} Mamba2 mixer layers")
    if total_layers == 0:
        raise RuntimeError("No Mamba2 mixer layers found")

    num_layers = resolve_num_layers(args.layers, total_layers)
    print(f"Permuting first {num_layers} layer(s)")

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    if args.determinism_check:
        print("Running determinism check with no permutation")
        logits_1 = compute_logits(model, input_ids)
        logits_2 = compute_logits(model, input_ids)
        det_metrics = summarize_logits_diff(logits_1, logits_2)
        print_metrics("[DETERMINISM] repeated forward", det_metrics)
        save_results_csv(
            [
                {
                    "tag": "determinism",
                    "model_name": MODEL_NAME,
                    "tokenizer_name": TOKENIZER_NAME,
                    "layers_arg": args.layers,
                    "num_layers": 0,
                    "negative_control": False,
                    "determinism_check": True,
                    "ablate": "",
                    "max_abs_diff": f"{det_metrics['max_abs_diff']:.8e}",
                    "mean_abs_diff": f"{det_metrics['mean_abs_diff']:.8e}",
                    "relative_max_error": f"{det_metrics['relative_max_error']:.8e}",
                    "top1_match_rate": f"{det_metrics['top1_match_rate']:.8f}",
                    "logits_abs_max": f"{det_metrics['logits_abs_max']:.8e}",
                    "threshold_max": "",
                    "threshold_mean": "",
                    "threshold_relative": "",
                    "threshold_top1": "",
                    "status": "PASS" if det_metrics['max_abs_diff'] <= 1e-6 else "WARN",
                }
            ],
            output_path,
        )
        return

    original_logits = compute_logits(model, input_ids)

    perms_by_layer = {}
    for layer_idx, mixer in mixers[:num_layers]:
        perms_by_layer[layer_idx] = torch.randperm(int(getattr(mixer, "nheads")), device=mixer.in_proj.weight.device)

    good_model = copy.deepcopy(model).eval()
    good_model = configure_mamba2_for_equivalence(good_model, disable_mem_eff_path=args.disable_mem_eff_path)
    for layer_idx, mixer in collect_mamba2_mixers(good_model)[:num_layers]:
        permute_mamba2_heads_(mixer, perms_by_layer[layer_idx], ablate=args.ablate)
        suffix = "" if args.ablate is None else f" ablate={args.ablate}"
        print(f"Permuted layer {layer_idx} with nheads={mixer.nheads} headdim={mixer.headdim}{suffix}")
    good_logits = compute_logits(good_model, input_ids)
    good_metrics = summarize_logits_diff(original_logits, good_logits)
    good_tag = "[GOOD] synchronized head permutation"
    if args.ablate is not None:
        good_tag += f" ({args.ablate})"
    print_metrics(good_tag, good_metrics)

    good_status = (
        good_metrics["max_abs_diff"] < THRESHOLD_MAX
        and good_metrics["mean_abs_diff"] < THRESHOLD_MEAN
        and good_metrics["relative_max_error"] < THRESHOLD_RELATIVE
        and good_metrics["top1_match_rate"] >= THRESHOLD_TOP1
    )
    if good_status:
        print("[PASS] Mamba2 head permutation equivalence verified.")
    else:
        print("[FAIL] Mamba2 head permutation equivalence check failed.")

    rows = [
        {
            "tag": "good",
            "model_name": MODEL_NAME,
            "tokenizer_name": TOKENIZER_NAME,
            "layers_arg": args.layers,
            "num_layers": num_layers,
            "negative_control": args.negative_control,
            "determinism_check": False,
            "ablate": args.ablate or "",
            "max_abs_diff": f"{good_metrics['max_abs_diff']:.8e}",
            "mean_abs_diff": f"{good_metrics['mean_abs_diff']:.8e}",
            "relative_max_error": f"{good_metrics['relative_max_error']:.8e}",
            "top1_match_rate": f"{good_metrics['top1_match_rate']:.8f}",
            "logits_abs_max": f"{good_metrics['logits_abs_max']:.8e}",
            "threshold_max": f"{THRESHOLD_MAX:.8e}",
            "threshold_mean": f"{THRESHOLD_MEAN:.8e}",
            "threshold_relative": f"{THRESHOLD_RELATIVE:.8e}",
            "threshold_top1": f"{THRESHOLD_TOP1:.8f}",
            "status": "PASS" if good_status else "FAIL",
        }
    ]

    if args.negative_control:
        bad_model = copy.deepcopy(model).eval()
        bad_model = configure_mamba2_for_equivalence(bad_model, disable_mem_eff_path=args.disable_mem_eff_path)
        for layer_idx, mixer in collect_mamba2_mixers(bad_model)[:num_layers]:
            bad_permute_mamba2_inproj_only_(mixer, perms_by_layer[layer_idx])
            print(f"Applied negative control on layer {layer_idx} with nheads={mixer.nheads}")
        bad_logits = compute_logits(bad_model, input_ids)
        bad_metrics = summarize_logits_diff(original_logits, bad_logits)
        print_metrics("[BAD] in_proj-only head permutation", bad_metrics)
        rows.append(
            {
                "tag": "bad",
                "model_name": MODEL_NAME,
                "tokenizer_name": TOKENIZER_NAME,
                "layers_arg": args.layers,
                "num_layers": num_layers,
                "negative_control": args.negative_control,
                "determinism_check": False,
                "ablate": args.ablate or "",
                "max_abs_diff": f"{bad_metrics['max_abs_diff']:.8e}",
                "mean_abs_diff": f"{bad_metrics['mean_abs_diff']:.8e}",
                "relative_max_error": f"{bad_metrics['relative_max_error']:.8e}",
                "top1_match_rate": f"{bad_metrics['top1_match_rate']:.8f}",
                "logits_abs_max": f"{bad_metrics['logits_abs_max']:.8e}",
                "threshold_max": "",
                "threshold_mean": "",
                "threshold_relative": "",
                "threshold_top1": "",
                "status": "OBSERVED",
            }
        )

    save_results_csv(rows, output_path)
    if not good_status:
        print(
            "[WARN] Good permutation did not meet the current strict thresholds: "
            f"max_abs_diff={good_metrics['max_abs_diff']:.8e}, "
            f"mean_abs_diff={good_metrics['mean_abs_diff']:.8e}, "
            f"relative_max_error={good_metrics['relative_max_error']:.8e}, "
            f"top1_match_rate={good_metrics['top1_match_rate']:.8f}"
        )


if __name__ == "__main__":
    main()
