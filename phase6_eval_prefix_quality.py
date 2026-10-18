import argparse
import copy
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from elastic_mamba1_block import ElasticMamba1Block
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules import mamba_simple
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops import selective_scan_interface
from phase1_stress_test_mamba1_permutation import permute_mamba1_expand_channels_
from output_naming import build_output_path, build_related_output_path


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


MODEL_NAME = "state-spaces/mamba-130m"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
SEED = 0
DEFAULT_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", nargs="+", default=["random", "magnitude", "grad_weight"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="fp32", choices=list(DEFAULT_DTYPES.keys()))
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def prepare_runtime():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        mamba_simple.causal_conv1d_fn = None
        mamba_simple.selective_scan_fn = selective_scan_interface.selective_scan_ref
    return device


def synchronize_if_needed(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def ratio_to_active_channels(d_inner: int, ratio: float, align_to: int = 8):
    active_channels = int(round(d_inner * ratio))
    active_channels = max(1, min(d_inner, active_channels))
    if align_to <= 1 or d_inner < align_to:
        return active_channels
    if active_channels == d_inner:
        return d_inner
    aligned = max(align_to, int(round(active_channels / align_to)) * align_to)
    if aligned > d_inner:
        aligned = (d_inner // align_to) * align_to
    return max(1, min(d_inner, aligned))


@torch.no_grad()
def collect_mamba_mixers(model: MambaLMHeadModel):
    mixers = []
    for layer_idx, layer in enumerate(model.backbone.layers):
        mixer = getattr(layer, "mixer", None)
        if isinstance(mixer, Mamba) or isinstance(mixer, ElasticMamba1Block):
            mixers.append((layer_idx, mixer))
    return mixers


@torch.no_grad()
def build_model(device: str, dtype: torch.dtype):
    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, device=device, dtype=dtype)
    model.eval()
    return model


@torch.no_grad()
def sample_validation_texts(tokenizer, num_samples: int, seq_len: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "`datasets` is required for phase6_eval_prefix_quality.py. "
            "Please install Hugging Face datasets in the active environment."
        ) from exc

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    tokenized = []
    for item in dataset:
        text = item["text"].strip()
        if not text:
            continue
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        input_ids = encoded["input_ids"][0]
        if input_ids.numel() < 2:
            continue
        tokenized.append(input_ids)
        if len(tokenized) >= num_samples:
            break
    if len(tokenized) < num_samples:
        raise ValueError(f"Only collected {len(tokenized)} usable samples, fewer than requested {num_samples}.")
    return tokenized


@torch.no_grad()
def compute_magnitude_perm(mixer):
    score = mixer.out_proj.weight[:, : mixer.d_inner].abs().sum(dim=0)
    return torch.argsort(score, descending=True)


def compute_grad_weight_perm(model, mixer_pairs, samples, device: str):
    model.zero_grad(set_to_none=True)
    for input_ids in samples:
        batch = input_ids.unsqueeze(0).to(device)
        logits = model(batch).logits[:, :-1, :]
        labels = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean")
        loss.backward()

    perms = []
    for _, mixer in mixer_pairs:
        grad = mixer.out_proj.weight.grad
        if grad is None:
            raise RuntimeError("Missing gradients for out_proj.weight; grad_weight ordering cannot be computed.")
        score = (grad[:, : mixer.d_inner] * mixer.out_proj.weight[:, : mixer.d_inner]).abs().sum(dim=0)
        perms.append(torch.argsort(score, descending=True))
    model.zero_grad(set_to_none=True)
    return perms


@torch.no_grad()
def compute_random_perm(mixer):
    return torch.randperm(mixer.d_inner, device=mixer.in_proj.weight.device)


def apply_ordering_to_model(model, order_name: str, calibration_samples, device: str):
    reordered_model = copy.deepcopy(model).eval()
    mixer_pairs = collect_mamba_mixers(reordered_model)

    if order_name == "random":
        perms = [compute_random_perm(mixer) for _, mixer in mixer_pairs]
    elif order_name == "magnitude":
        perms = [compute_magnitude_perm(mixer).to(mixer.in_proj.weight.device) for _, mixer in mixer_pairs]
    elif order_name == "grad_weight":
        with torch.enable_grad():
            perms = compute_grad_weight_perm(reordered_model, mixer_pairs, calibration_samples, device)
    else:
        raise ValueError(f"Unsupported ordering: {order_name}")

    for (_, mixer), perm in zip(mixer_pairs, perms):
        permute_mamba1_expand_channels_(mixer, perm)
    return reordered_model, perms


@torch.no_grad()
def replace_mixers_with_elastic(model, active_channels: int, force_prefix_path: bool = True):
    for layer in model.backbone.layers:
        mixer = getattr(layer, "mixer", None)
        if isinstance(mixer, Mamba):
            elastic = ElasticMamba1Block.from_mamba(mixer).eval()
            elastic.set_active_channels(active_channels, force_prefix_path=force_prefix_path)
            layer.mixer = elastic
        elif isinstance(mixer, ElasticMamba1Block):
            mixer.set_active_channels(active_channels, force_prefix_path=force_prefix_path)
    return model


@torch.no_grad()
def compute_latency_ms(model, input_ids, device: str):
    synchronize_if_needed(device)
    start = time.perf_counter()
    _ = model(input_ids)
    synchronize_if_needed(device)
    end = time.perf_counter()
    return (end - start) * 1000.0


@torch.no_grad()
def evaluate_model(model, full_model, samples, device: str):
    total_loss = 0.0
    total_tokens = 0
    total_logits_mse = 0.0
    total_latency_ms = 0.0

    for input_ids in samples:
        batch = input_ids.unsqueeze(0).to(device)
        logits = model(batch).logits
        full_logits = full_model(batch).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        token_count = shift_labels.numel()
        total_loss += loss.item()
        total_tokens += token_count
        total_logits_mse += torch.mean((logits.float() - full_logits.float()) ** 2).item()
        total_latency_ms += compute_latency_ms(model, batch, device)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    avg_logits_mse = total_logits_mse / len(samples)
    avg_latency_ms = total_latency_ms / len(samples)
    return {
        "next_token_loss": avg_loss,
        "perplexity": perplexity,
        "logits_mse_to_full": avg_logits_mse,
        "average_latency_ms": avg_latency_ms,
    }


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = prepare_runtime()
    dtype = DEFAULT_DTYPES[args.dtype]

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    base_model = build_model(device=device, dtype=dtype)
    mixer_pairs = collect_mamba_mixers(base_model)
    if len(mixer_pairs) != 24:
        print(f"[WARN] Expected 24 Mamba mixers, found {len(mixer_pairs)}")

    print("Loading and tokenizing WikiText-2 validation samples")
    samples = sample_validation_texts(tokenizer, num_samples=args.num_samples, seq_len=args.seq_len)
    calibration_count = min(8, len(samples))
    calibration_samples = samples[:calibration_count]

    csv_path = build_output_path(
        args.output_csv,
        "phase6_prefix_quality",
        tags={
            "dtype": args.dtype,
            "orders": "-".join(args.orders),
            "ratios": "-".join(f"{ratio:.2f}" for ratio in args.ratios),
            "num_samples": args.num_samples,
            "seq_len": args.seq_len,
        },
    )
    high_ratio_csv_path = build_related_output_path(csv_path, "high_ratios")

    rows = []
    summary_by_ratio = {}
    for order_name in args.orders:
        print(f"Applying ordering: {order_name}")
        ordered_model, _ = apply_ordering_to_model(base_model, order_name, calibration_samples, device)
        full_model = replace_mixers_with_elastic(copy.deepcopy(ordered_model).eval(), active_channels=None, force_prefix_path=False)

        full_metrics = evaluate_model(full_model, full_model, samples, device)
        full_d_inner = collect_mamba_mixers(full_model)[0][1].d_inner

        for ratio in args.ratios:
            active_channels = ratio_to_active_channels(full_d_inner, ratio)
            ratio_model = replace_mixers_with_elastic(copy.deepcopy(ordered_model).eval(), active_channels=active_channels, force_prefix_path=True)
            metrics = evaluate_model(ratio_model, full_model, samples, device)
            print(
                f"order={order_name} ratio={ratio:.2f} k={active_channels} "
                f"loss={metrics['next_token_loss']:.6f} ppl={metrics['perplexity']:.6f} "
                f"logits_mse={metrics['logits_mse_to_full']:.8e} latency_ms={metrics['average_latency_ms']:.4f}"
            )
            summary_by_ratio.setdefault(ratio, {})[order_name] = metrics
            rows.append(
                {
                    "order": order_name,
                    "ratio": f"{ratio:.2f}",
                    "active_channels": active_channels,
                    "num_layers": len(mixer_pairs),
                    "num_samples": args.num_samples,
                    "seq_len": args.seq_len,
                    "dtype": args.dtype,
                    "next_token_loss": f"{metrics['next_token_loss']:.8f}",
                    "perplexity": f"{metrics['perplexity']:.8f}",
                    "logits_mse_to_full": f"{metrics['logits_mse_to_full']:.8e}",
                    "average_latency_ms": f"{metrics['average_latency_ms']:.6f}",
                    "full_loss": f"{full_metrics['next_token_loss']:.8f}",
                    "full_perplexity": f"{full_metrics['perplexity']:.8f}",
                    "full_average_latency_ms": f"{full_metrics['average_latency_ms']:.6f}",
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "order",
                "ratio",
                "active_channels",
                "num_layers",
                "num_samples",
                "seq_len",
                "dtype",
                "next_token_loss",
                "perplexity",
                "logits_mse_to_full",
                "average_latency_ms",
                "full_loss",
                "full_perplexity",
                "full_average_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    high_ratio_rows = []
    print("\nSummary by ratio")
    for ratio in sorted(summary_by_ratio):
        print(f"ratio={ratio:.2f}")
        ratio_metrics = summary_by_ratio[ratio]
        for order_name in args.orders:
            metrics = ratio_metrics.get(order_name)
            if metrics is None:
                continue
            print(
                f"  order={order_name:<11} loss={metrics['next_token_loss']:.6f} "
                f"ppl={metrics['perplexity']:.6f} latency_ms={metrics['average_latency_ms']:.4f}"
            )
        if "grad_weight" in ratio_metrics and "random" in ratio_metrics:
            loss_reduction = ratio_metrics["random"]["next_token_loss"] - ratio_metrics["grad_weight"]["next_token_loss"]
            print(f"  grad_weight_loss_reduction_vs_random={loss_reduction:.6f}")
        else:
            loss_reduction = None

        for order_name in args.orders:
            metrics = ratio_metrics.get(order_name)
            if metrics is None:
                continue
            high_ratio_rows.append(
                {
                    "ratio": f"{ratio:.2f}",
                    "order": order_name,
                    "next_token_loss": f"{metrics['next_token_loss']:.8f}",
                    "perplexity": f"{metrics['perplexity']:.8f}",
                    "average_latency_ms": f"{metrics['average_latency_ms']:.6f}",
                    "grad_weight_loss_reduction_vs_random": "" if loss_reduction is None else f"{loss_reduction:.8f}",
                }
            )

    with high_ratio_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ratio",
                "order",
                "next_token_loss",
                "perplexity",
                "average_latency_ms",
                "grad_weight_loss_reduction_vs_random",
            ],
        )
        writer.writeheader()
        writer.writerows(high_ratio_rows)

    print(f"Saved results to {csv_path}")
    print(f"Saved high-ratio summary to {high_ratio_csv_path}")


if __name__ == "__main__":
    main()
