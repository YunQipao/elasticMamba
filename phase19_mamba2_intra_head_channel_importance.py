import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mamba2_permutation_utils import get_mamba2_layout
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--calib_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def sample_wikitext2_validation_texts(tokenizer, num_samples: int, seq_len: int, seed: int, local_files_only: bool = False):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "`datasets` is required for phase19_mamba2_intra_head_channel_importance.py. "
            "Please install Hugging Face datasets in the active environment."
        ) from exc

    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="validation",
        download_mode="reuse_dataset_if_exists",
    )
    candidate_indices = [idx for idx, item in enumerate(dataset) if item["text"].strip()]
    if len(candidate_indices) < num_samples:
        raise ValueError(f"Only found {len(candidate_indices)} non-empty validation texts, fewer than requested {num_samples}.")

    random.Random(seed).shuffle(candidate_indices)
    tokenized = []
    for idx in candidate_indices:
        encoded = tokenizer(dataset[idx]["text"], return_tensors="pt", truncation=True, max_length=seq_len)
        input_ids = encoded["input_ids"][0]
        if input_ids.numel() < 2:
            continue
        tokenized.append(input_ids)
        if len(tokenized) >= num_samples:
            break
    if len(tokenized) < num_samples:
        raise ValueError(f"Only collected {len(tokenized)} usable samples, fewer than requested {num_samples}.")
    return tokenized


def _safe_mul_abs_sum(tensor, grad, reduce_dims=None):
    if tensor is None or grad is None:
        return None
    value = (tensor.float() * grad.float()).abs()
    if reduce_dims is None:
        return value
    return value.sum(dim=reduce_dims)


def _gini_from_probs(probs: torch.Tensor):
    if probs.numel() == 0:
        return 0.0
    sorted_probs, _ = torch.sort(probs.float())
    n = sorted_probs.numel()
    total = sorted_probs.sum().item()
    if total <= 0:
        return 0.0
    weighted_sum = torch.sum(torch.arange(1, n + 1, dtype=torch.float32) * sorted_probs.cpu()).item()
    return (2.0 * weighted_sum) / (n * total) - (n + 1) / n


def _entropy_from_probs(probs: torch.Tensor):
    probs = probs.float()
    total = probs.sum().item()
    if total <= 0:
        return 0.0
    normalized = probs / total
    normalized = normalized[normalized > 0]
    if normalized.numel() == 0:
        return 0.0
    return float((-(normalized * normalized.log()).sum()).item())


def _add_rows_channel_scores(scores, weight, grad, nheads, headdim):
    contribution = _safe_mul_abs_sum(weight, grad, reduce_dims=tuple(range(1, weight.dim())))
    if contribution is not None:
        scores += contribution.reshape(nheads, headdim)


def _add_vector_channel_scores(scores, vector, grad, nheads, headdim):
    contribution = _safe_mul_abs_sum(vector, grad)
    if contribution is not None:
        scores += contribution.reshape(nheads, headdim)


def _add_linear_rows_slice_channel_scores(scores, linear, start: int, end: int, nheads: int, headdim: int):
    weight_grad = getattr(linear.weight, "grad", None)
    if weight_grad is not None:
        _add_rows_channel_scores(scores, linear.weight[start:end, :], weight_grad[start:end, :], nheads, headdim)


def _add_conv1d_channel_slice_scores(scores, conv1d, start: int, end: int, nheads: int, headdim: int):
    weight_grad = getattr(conv1d.weight, "grad", None)
    if weight_grad is not None:
        _add_rows_channel_scores(scores, conv1d.weight[start:end, :, :], weight_grad[start:end, :, :], nheads, headdim)


def _add_bias_slice_channel_scores(scores, bias, start: int, end: int, nheads: int, headdim: int):
    if bias is None:
        return
    bias_grad = getattr(bias, "grad", None)
    if bias_grad is not None:
        _add_vector_channel_scores(scores, bias[start:end], bias_grad[start:end], nheads, headdim)


def compute_intra_head_channel_scores(model, mixer_pairs, calibration_samples, device: str):
    model.zero_grad(set_to_none=True)
    num_calibration_samples = len(calibration_samples)
    for input_ids in calibration_samples:
        batch = input_ids.unsqueeze(0).to(device)
        logits = model(batch).logits[:, :-1, :]
        labels = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean")
        (loss / num_calibration_samples).backward()

    all_scores = []
    for layer_idx, mixer in mixer_pairs:
        layout = get_mamba2_layout(mixer)
        nheads = int(layout["nheads"])
        headdim = int(layout["headdim"])
        d_ssm = int(layout["d_ssm"])
        scores = torch.zeros(nheads, headdim, dtype=torch.float32, device=mixer.in_proj.weight.device)

        z_start, z_end = layout["in_proj"]["z"]
        x_start, x_end = layout["in_proj"]["x"]
        conv_x_start, conv_x_end = layout["conv1d"]["x"]

        _add_linear_rows_slice_channel_scores(scores, mixer.in_proj, z_start, z_end, nheads, headdim)
        _add_linear_rows_slice_channel_scores(scores, mixer.in_proj, x_start, x_end, nheads, headdim)

        _add_bias_slice_channel_scores(scores, mixer.in_proj.bias, z_start, z_end, nheads, headdim)
        _add_bias_slice_channel_scores(scores, mixer.in_proj.bias, x_start, x_end, nheads, headdim)

        _add_conv1d_channel_slice_scores(scores, mixer.conv1d, conv_x_start, conv_x_end, nheads, headdim)
        _add_bias_slice_channel_scores(scores, mixer.conv1d.bias, conv_x_start, conv_x_end, nheads, headdim)

        if getattr(mixer, "norm", None) is not None and getattr(mixer.norm, "weight", None) is not None:
            _add_vector_channel_scores(scores, mixer.norm.weight, mixer.norm.weight.grad, nheads, headdim)

        out_proj_weight_grad = getattr(mixer.out_proj.weight, "grad", None)
        out_proj_contribution = _safe_mul_abs_sum(
            mixer.out_proj.weight[:, :d_ssm],
            None if out_proj_weight_grad is None else out_proj_weight_grad[:, :d_ssm],
            reduce_dims=(0,),
        )
        if out_proj_contribution is not None:
            scores += out_proj_contribution.reshape(nheads, headdim)

        all_scores.append((layer_idx, scores.detach().float().cpu()))

    model.zero_grad(set_to_none=True)
    return all_scores


def build_rows(score_tuples):
    detail_rows = []
    concentration_rows = []

    for layer_idx, scores in score_tuples:
        nheads, headdim = scores.shape
        for head_id in range(nheads):
            head_scores = scores[head_id]
            total = head_scores.sum().item()
            if total > 0:
                normalized = head_scores / total
            else:
                normalized = torch.zeros_like(head_scores)

            sorted_indices = torch.argsort(head_scores, descending=True)
            ranks = torch.empty_like(sorted_indices)
            ranks[sorted_indices] = torch.arange(1, headdim + 1, dtype=sorted_indices.dtype)

            top8_share = normalized[sorted_indices[:8]].sum().item()
            top16_share = normalized[sorted_indices[:16]].sum().item()
            top32_share = normalized[sorted_indices[:32]].sum().item()
            entropy = _entropy_from_probs(normalized)
            gini = _gini_from_probs(normalized)

            concentration_rows.append(
                {
                    "layer_idx": layer_idx,
                    "head_id": head_id,
                    "top8_share": f"{top8_share:.8f}",
                    "top16_share": f"{top16_share:.8f}",
                    "top32_share": f"{top32_share:.8f}",
                    "entropy": f"{entropy:.8f}",
                    "gini": f"{gini:.8f}",
                    "total_importance": f"{total:.8e}",
                }
            )

            for channel_id in range(headdim):
                detail_rows.append(
                    {
                        "layer_idx": layer_idx,
                        "head_id": head_id,
                        "channel_id": channel_id,
                        "importance": f"{head_scores[channel_id].item():.8e}",
                        "rank_in_head": int(ranks[channel_id].item()),
                        "normalized_importance": f"{normalized[channel_id].item():.8f}",
                    }
                )

    return detail_rows, concentration_rows


def print_summary(concentration_rows):
    top8_values = [float(row["top8_share"]) for row in concentration_rows]
    top16_values = [float(row["top16_share"]) for row in concentration_rows]
    top32_values = [float(row["top32_share"]) for row in concentration_rows]

    print("Summary")
    print(f"average top8_share  = {sum(top8_values) / len(top8_values):.8f}")
    print(f"average top16_share = {sum(top16_values) / len(top16_values):.8f}")
    print(f"average top32_share = {sum(top32_values) / len(top32_values):.8f}")

    most_concentrated = max(concentration_rows, key=lambda row: float(row["top8_share"]))
    most_dispersed = min(concentration_rows, key=lambda row: float(row["top8_share"]))
    print(
        "most concentrated head = "
        f"layer {most_concentrated['layer_idx']} head {most_concentrated['head_id']} "
        f"top8_share={float(most_concentrated['top8_share']):.8f} "
        f"entropy={float(most_concentrated['entropy']):.8f}"
    )
    print(
        "most dispersed head   = "
        f"layer {most_dispersed['layer_idx']} head {most_dispersed['head_id']} "
        f"top8_share={float(most_dispersed['top8_share']):.8f} "
        f"entropy={float(most_dispersed['entropy']):.8f}"
    )

    layer_scores = {}
    for row in concentration_rows:
        layer_idx = int(row["layer_idx"])
        layer_scores.setdefault(layer_idx, []).append(float(row["top8_share"]))
    sorted_layers = sorted(
        ((layer_idx, sum(values) / len(values)) for layer_idx, values in layer_scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    print("layer concentration ranking by average top8_share")
    for layer_idx, avg_top8 in sorted_layers:
        print(f"  layer={layer_idx:02d} avg_top8_share={avg_top8:.8f}")


def save_csv(rows, path: Path, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_local_hf_cache(output_dir)
    detail_csv_path = output_dir / "phase19_mamba2_intra_head_channel_importance.csv"
    concentration_csv_path = output_dir / "phase19_mamba2_intra_head_channel_concentration.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    model, loader_name = build_model(device=device, dtype=dtype)
    print(f"Model loader: {loader_name}")
    model = configure_mamba2_for_equivalence(model, disable_mem_eff_path=True)
    mixer_pairs = collect_mamba2_mixers(model)
    if not mixer_pairs:
        raise RuntimeError("No Mamba2 mixer layers found")

    print("Loading and tokenizing WikiText-2 validation samples")
    samples = sample_wikitext2_validation_texts(
        tokenizer,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    if args.calib_samples < 1 or args.calib_samples > len(samples):
        raise ValueError(f"--calib_samples must be in [1, {len(samples)}], got {args.calib_samples}")

    shuffled_indices = list(range(len(samples)))
    random.Random(args.seed).shuffle(shuffled_indices)
    calibration_samples = [samples[idx] for idx in shuffled_indices[: args.calib_samples]]
    print(f"Using {args.calib_samples} calibration samples from {args.num_samples} sampled validation texts")

    with torch.enable_grad():
        score_tuples = compute_intra_head_channel_scores(model, mixer_pairs, calibration_samples, device)

    detail_rows, concentration_rows = build_rows(score_tuples)
    save_csv(
        detail_rows,
        detail_csv_path,
        fieldnames=["layer_idx", "head_id", "channel_id", "importance", "rank_in_head", "normalized_importance"],
    )
    save_csv(
        concentration_rows,
        concentration_csv_path,
        fieldnames=["layer_idx", "head_id", "top8_share", "top16_share", "top32_share", "entropy", "gini", "total_importance"],
    )

    print(f"Saved channel importance to {detail_csv_path}")
    print(f"Saved concentration summary to {concentration_csv_path}")
    print_summary(concentration_rows)
    model.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
