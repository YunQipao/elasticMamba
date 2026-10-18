import argparse
import copy
import csv
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from output_naming import build_output_path

from elastic_mamba2_head_block import ElasticMamba2HeadBlock
from mamba2_permutation_utils import get_mamba2_layout
from phase6_eval_prefix_quality import sample_validation_texts
from phase12_mamba2_head_permutation_equivalence import (
    MODEL_NAME,
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


SEED = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", nargs="+", default=["random", "grad_weight"])
    parser.add_argument("--active_heads", nargs="+", type=int, default=[16, 18, 20, 22, 24])
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def synchronize_if_needed(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def replace_mixers_with_elastic(model, layer_active_heads_map=None):
    if layer_active_heads_map is None:
        layer_active_heads_map = {}
    elastic_model = copy.deepcopy(model).eval()
    for layer_idx, layer in enumerate(elastic_model.backbone.layers):
        mixer = getattr(layer, "mixer", None)
        if mixer is not None and mixer.__class__.__name__ == "Mamba2":
            elastic = ElasticMamba2HeadBlock.from_mamba2(mixer).eval()
            elastic.set_active_head_indices(layer_active_heads_map.get(layer_idx))
            layer.mixer = elastic
        elif isinstance(mixer, ElasticMamba2HeadBlock):
            mixer.set_active_head_indices(layer_active_heads_map.get(layer_idx))
    return elastic_model


def _score_rows_by_head(weight, grad, nheads, headdim):
    return (weight.float() * grad.float()).abs().reshape(nheads, headdim, -1).sum(dim=(1, 2))


def _score_vec_by_head(vector, grad, nheads, headdim=None):
    value = (vector.float() * grad.float()).abs()
    if value.numel() == nheads:
        return value
    if headdim is None:
        raise ValueError("headdim is required for channel-wise vector scoring")
    return value.reshape(nheads, headdim).sum(dim=1)


def compute_grad_weight_head_orders(model, mixer_pairs, samples, device: str):
    model.zero_grad(set_to_none=True)
    for input_ids in samples:
        batch = input_ids.unsqueeze(0).to(device)
        logits = model(batch).logits[:, :-1, :]
        labels = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean")
        loss.backward()

    orders = []
    for _, mixer in mixer_pairs:
        layout = get_mamba2_layout(mixer)
        nheads = mixer.nheads
        headdim = mixer.headdim
        scores = torch.zeros(nheads, device=mixer.in_proj.weight.device, dtype=torch.float32)

        z_start, z_end = layout["in_proj"]["z"]
        x_start, x_end = layout["in_proj"]["x"]
        dt_start, dt_end = layout["in_proj"]["dt"]
        conv_x_start, conv_x_end = layout["conv1d"]["x"]

        scores += _score_rows_by_head(
            mixer.in_proj.weight[z_start:z_end, :],
            mixer.in_proj.weight.grad[z_start:z_end, :],
            nheads,
            headdim,
        )
        scores += _score_rows_by_head(
            mixer.in_proj.weight[x_start:x_end, :],
            mixer.in_proj.weight.grad[x_start:x_end, :],
            nheads,
            headdim,
        )
        scores += (mixer.in_proj.weight[dt_start:dt_end, :].float() * mixer.in_proj.weight.grad[dt_start:dt_end, :].float()).abs().sum(dim=1)
        if mixer.in_proj.bias is not None:
            scores += _score_vec_by_head(mixer.in_proj.bias[z_start:z_end], mixer.in_proj.bias.grad[z_start:z_end], nheads, headdim)
            scores += _score_vec_by_head(mixer.in_proj.bias[x_start:x_end], mixer.in_proj.bias.grad[x_start:x_end], nheads, headdim)
            scores += (mixer.in_proj.bias[dt_start:dt_end].float() * mixer.in_proj.bias.grad[dt_start:dt_end].float()).abs()

        scores += _score_rows_by_head(
            mixer.conv1d.weight[conv_x_start:conv_x_end, :, :],
            mixer.conv1d.weight.grad[conv_x_start:conv_x_end, :, :],
            nheads,
            headdim,
        )
        if mixer.conv1d.bias is not None:
            scores += _score_vec_by_head(mixer.conv1d.bias[conv_x_start:conv_x_end], mixer.conv1d.bias.grad[conv_x_start:conv_x_end], nheads, headdim)

        scores += _score_vec_by_head(mixer.A_log, mixer.A_log.grad, nheads)
        scores += _score_vec_by_head(mixer.dt_bias, mixer.dt_bias.grad, nheads)
        scores += _score_vec_by_head(mixer.D, mixer.D.grad, nheads, headdim if mixer.D.numel() != nheads else None)

        if getattr(mixer, "norm", None) is not None and getattr(mixer.norm, "weight", None) is not None:
            scores += _score_vec_by_head(mixer.norm.weight, mixer.norm.weight.grad, nheads, headdim)
        scores += (mixer.out_proj.weight.float() * mixer.out_proj.weight.grad.float()).abs().reshape(mixer.out_proj.weight.shape[0], nheads, headdim).sum(dim=(0, 2))

        orders.append(torch.argsort(scores, descending=True))

    model.zero_grad(set_to_none=True)
    return orders


@torch.no_grad()
def build_head_orders(model, order_name, calibration_samples, device: str):
    mixer_pairs = collect_mamba2_mixers(model)
    if order_name == "random":
        generator = torch.Generator(device=device if device == "cuda" else "cpu")
        generator.manual_seed(SEED)
        return [torch.randperm(mixer.nheads, generator=generator, device=mixer.in_proj.weight.device) for _, mixer in mixer_pairs]
    if order_name == "grad_weight":
        with torch.enable_grad():
            return compute_grad_weight_head_orders(model, mixer_pairs, calibration_samples, device)
    raise ValueError(f"Unsupported order: {order_name}")


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
    total_top1_match = 0.0
    total_positions = 0
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
        top1 = logits.argmax(dim=-1)
        full_top1 = full_logits.argmax(dim=-1)
        total_top1_match += (top1 == full_top1).float().sum().item()
        total_positions += top1.numel()
        total_latency_ms += compute_latency_ms(model, batch, device)

    avg_loss = total_loss / total_tokens
    return {
        "next_token_loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "logits_mse_to_full": total_logits_mse / len(samples),
        "top1_agreement_to_full": total_top1_match / total_positions,
        "average_latency_ms": total_latency_ms / len(samples),
    }


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    csv_path = build_output_path(
        args.output_csv,
        "phase13_mamba2_logical_head_prefix_quality",
        tags={
            "orders": "-".join(args.orders),
            "active_heads": "-".join(map(str, args.active_heads)),
            "num_samples": args.num_samples,
            "seq_len": args.seq_len,
        },
    )
    ensure_local_hf_cache(csv_path.parent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    base_model, loader_name = build_model(device=device, dtype=dtype)
    print(f"Model loader: {loader_name}")
    base_model = configure_mamba2_for_equivalence(base_model, disable_mem_eff_path=True)
    mixer_pairs = collect_mamba2_mixers(base_model)
    if len(mixer_pairs) != 24:
        print(f"[WARN] Expected 24 Mamba2 mixers, found {len(mixer_pairs)}")

    print("Loading and tokenizing WikiText-2 validation samples")
    samples = sample_validation_texts(tokenizer, num_samples=args.num_samples, seq_len=args.seq_len)
    calibration_samples = samples[: min(8, len(samples))]

    full_model = replace_mixers_with_elastic(base_model, layer_active_heads_map={})
    full_metrics = evaluate_model(full_model, full_model, samples, device)
    print(
        f"full loss={full_metrics['next_token_loss']:.6f} ppl={full_metrics['perplexity']:.6f} "
        f"latency_ms={full_metrics['average_latency_ms']:.4f}"
    )

    rows = []
    for order_name in args.orders:
        print(f"Applying head order: {order_name}")
        ordered_heads = build_head_orders(copy.deepcopy(base_model).eval(), order_name, calibration_samples, device)

        for active_heads in args.active_heads:
            layer_active_heads_map = {}
            for (layer_idx, mixer), order in zip(mixer_pairs, ordered_heads):
                if active_heads > mixer.nheads:
                    raise ValueError(f"active_heads={active_heads} exceeds nheads={mixer.nheads}")
                selected = torch.sort(order[:active_heads]).values.tolist()
                layer_active_heads_map[layer_idx] = selected

            ratio_model = replace_mixers_with_elastic(base_model, layer_active_heads_map=layer_active_heads_map)
            metrics = evaluate_model(ratio_model, full_model, samples, device)
            print(
                f"order={order_name} active_heads={active_heads} "
                f"loss={metrics['next_token_loss']:.6f} ppl={metrics['perplexity']:.6f} "
                f"logits_mse={metrics['logits_mse_to_full']:.8e} "
                f"top1_agree={metrics['top1_agreement_to_full']:.8f} "
                f"latency_ms={metrics['average_latency_ms']:.4f}"
            )
            rows.append(
                {
                    "order": order_name,
                    "active_heads": active_heads,
                    "num_layers": len(mixer_pairs),
                    "num_samples": args.num_samples,
                    "seq_len": args.seq_len,
                    "next_token_loss": f"{metrics['next_token_loss']:.8f}",
                    "perplexity": f"{metrics['perplexity']:.8f}",
                    "logits_mse_to_full": f"{metrics['logits_mse_to_full']:.8e}",
                    "top1_agreement_to_full": f"{metrics['top1_agreement_to_full']:.8f}",
                    "average_latency_ms": f"{metrics['average_latency_ms']:.6f}",
                    "full_loss": f"{full_metrics['next_token_loss']:.8f}",
                    "full_perplexity": f"{full_metrics['perplexity']:.8f}",
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "order",
                "active_heads",
                "num_layers",
                "num_samples",
                "seq_len",
                "next_token_loss",
                "perplexity",
                "logits_mse_to_full",
                "top1_agreement_to_full",
                "average_latency_ms",
                "full_loss",
                "full_perplexity",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
