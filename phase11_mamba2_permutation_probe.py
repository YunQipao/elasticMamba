import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


MODEL_NAME = "state-spaces/mamba2-130m"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
DEFAULT_PROMPT = (
    "Mamba2 permutation probe: summarize the internal mixer structure and "
    "confirm a forward pass produces logits."
)
CANDIDATE_ATTRS = [
    "in_proj",
    "conv1d",
    "x_proj",
    "dt_proj",
    "bc_proj",
    "B_proj",
    "C_proj",
    "A_log",
    "A",
    "D",
    "out_proj",
]
CANDIDATE_PROPS = [
    "d_model",
    "d_inner",
    "d_state",
    "d_conv",
    "expand",
    "dt_rank",
    "nheads",
    "headdim",
    "ngroups",
    "chunk_size",
    "head_dim",
    "num_heads",
]


def ensure_local_hf_cache(output_dir: Path):
    cache_root = output_dir / "hf_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))


def tensor_summary(name, tensor):
    shape = tuple(tensor.shape)
    return f"{name}: shape={shape} dtype={tensor.dtype} device={tensor.device}"


def module_summary(name, module):
    parts = [f"{name}: type={module.__class__.__name__}"]
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    if isinstance(weight, torch.Tensor):
        parts.append(f"weight={tuple(weight.shape)}")
    if isinstance(bias, torch.Tensor):
        parts.append(f"bias={tuple(bias.shape)}")
    if isinstance(module, nn.Conv1d):
        parts.append(
            f"kernel_size={module.kernel_size} stride={module.stride} padding={module.padding} groups={module.groups}"
        )
    return " ".join(parts)


def attr_summary(name, value):
    if value is None:
        return f"{name}: None"
    if isinstance(value, (torch.nn.Parameter, torch.Tensor)):
        return tensor_summary(name, value)
    if isinstance(value, nn.Module):
        return module_summary(name, value)
    return f"{name}: type={type(value).__name__} value={value}"


def collect_mixer_layers(model):
    mixers = []
    if hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
        for layer_idx, layer in enumerate(model.backbone.layers):
            mixer = getattr(layer, "mixer", None)
            if mixer is not None:
                mixers.append((layer_idx, f"backbone.layers.{layer_idx}.mixer", mixer))
        if mixers:
            return mixers

    for name, module in model.named_modules():
        if name.endswith(".mixer"):
            layer_idx = len(mixers)
            mixers.append((layer_idx, name, module))
    return mixers


def render_mixer_report(layer_idx, mixer_name, mixer):
    lines = []
    lines.append(f"[Layer {layer_idx}] {mixer_name}")
    lines.append(f"mixer_type: {mixer.__class__.__module__}.{mixer.__class__.__name__}")

    prop_parts = []
    for prop in CANDIDATE_PROPS:
        if hasattr(mixer, prop):
            try:
                prop_parts.append(f"{prop}={getattr(mixer, prop)}")
            except Exception as exc:
                prop_parts.append(f"{prop}=<error:{exc}>")
    if prop_parts:
        lines.append("properties: " + ", ".join(prop_parts))
    else:
        lines.append("properties: <none of the requested attributes found>")

    lines.append("candidate_attrs:")
    for attr in CANDIDATE_ATTRS:
        if hasattr(mixer, attr):
            try:
                lines.append("  - " + attr_summary(attr, getattr(mixer, attr)))
            except Exception as exc:
                lines.append(f"  - {attr}: <error reading attribute: {exc}>")
        else:
            lines.append(f"  - {attr}: <missing>")

    lines.append("direct_child_modules:")
    child_items = list(mixer.named_children())
    if child_items:
        for child_name, child in child_items:
            lines.append("  - " + module_summary(child_name, child))
    else:
        lines.append("  - <none>")

    lines.append("direct_parameters:")
    direct_params = list(mixer.named_parameters(recurse=False))
    if direct_params:
        for param_name, param in direct_params:
            lines.append("  - " + tensor_summary(param_name, param))
    else:
        lines.append("  - <none>")

    lines.append("direct_buffers:")
    direct_buffers = list(mixer.named_buffers(recurse=False))
    if direct_buffers:
        for buffer_name, buffer in direct_buffers:
            lines.append("  - " + tensor_summary(buffer_name, buffer))
    else:
        lines.append("  - <none>")

    lines.append("selected_named_parameters:")
    selected = []
    for param_name, param in mixer.named_parameters():
        if any(token in param_name for token in ["in_proj", "conv", "x_proj", "dt", "bc", "out_proj", "A", "D"]):
            selected.append((param_name, param))
    if selected:
        for param_name, param in selected:
            lines.append("  - " + tensor_summary(param_name, param))
    else:
        lines.append("  - <none matched filter>")

    lines.append("")
    return lines


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
def run_forward_sanity(model, tokenizer, device: str):
    encoded = tokenizer(DEFAULT_PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    outputs = model(input_ids)
    logits = outputs.logits
    result = {
        "input_shape": tuple(input_ids.shape),
        "logits_shape": tuple(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_abs_max": float(logits.detach().abs().max().item()),
    }
    return result


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_local_hf_cache(output_dir)
    report_path = output_dir / "phase11_mamba2_structure.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    model, loader_name = build_model(device=device, dtype=dtype)
    print(f"Model loader: {loader_name}")

    mixers = collect_mixer_layers(model)
    print(f"Found {len(mixers)} mixer layers")

    report_lines = []
    report_lines.append(f"model_name: {MODEL_NAME}")
    report_lines.append(f"tokenizer_name: {TOKENIZER_NAME}")
    report_lines.append(f"device: {device}")
    report_lines.append(f"dtype: {dtype}")
    report_lines.append(f"model_loader: {loader_name}")
    report_lines.append(f"num_mixers: {len(mixers)}")
    report_lines.append("")

    for layer_idx, mixer_name, mixer in mixers:
        lines = render_mixer_report(layer_idx, mixer_name, mixer)
        report_lines.extend(lines)
        print(f"[Layer {layer_idx}] {mixer_name} type={mixer.__class__.__name__}")

    sanity = run_forward_sanity(model, tokenizer, device)
    sanity_lines = [
        "forward_sanity:",
        f"  input_shape={sanity['input_shape']}",
        f"  logits_shape={sanity['logits_shape']}",
        f"  logits_dtype={sanity['logits_dtype']}",
        f"  logits_abs_max={sanity['logits_abs_max']:.8e}",
    ]
    report_lines.extend(sanity_lines)
    print("Forward sanity check")
    for line in sanity_lines[1:]:
        print(line.strip())

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved structure report to {report_path}")


if __name__ == "__main__":
    main()
