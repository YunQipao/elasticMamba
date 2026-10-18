import argparse
import csv
import time
from pathlib import Path

import torch
from mamba_ssm import Mamba
from mamba_ssm.modules import mamba_simple
from mamba_ssm.ops import selective_scan_interface

from elastic_mamba1_block import ElasticMamba1Block


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=768)
    return parser.parse_args()


def build_mamba(d_model: int, d_state: int, expand: int, device: str, dtype: torch.dtype):
    try:
        module = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=expand,
            use_fast_path=False,
            device=device,
            dtype=dtype,
        )
    except TypeError:
        module = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=expand,
            device=device,
            dtype=dtype,
        )
        if hasattr(module, "use_fast_path"):
            module.use_fast_path = False
    module.eval()
    return module


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
def measure_latency_stats_ms(fn, warmup: int, iters: int, device: str):
    for _ in range(warmup):
        synchronize_if_needed(device)
        _ = fn()
        synchronize_if_needed(device)

    latencies = []
    for _ in range(iters):
        synchronize_if_needed(device)
        start = time.perf_counter()
        _ = fn()
        synchronize_if_needed(device)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    values = torch.tensor(latencies, dtype=torch.float64)
    return {
        "mean_latency_ms": values.mean().item(),
        "median_latency_ms": values.median().item(),
        "p95_latency_ms": torch.quantile(values, 0.95).item(),
    }


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = prepare_runtime()
    dtype = torch.float32
    d_state = 16
    expand = 2
    warmup = 30
    iters = 100

    base = build_mamba(d_model=args.d_model, d_state=d_state, expand=expand, device=device, dtype=dtype)
    elastic = ElasticMamba1Block.from_mamba(base).eval()

    x = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device, dtype=dtype)

    y_base = base(x)
    y_elastic_full = elastic(x, active_channels=None)
    full_equiv_diff = (y_base - y_elastic_full).abs().max().item()
    print(f"active_channels=None vs base max_abs_diff = {full_equiv_diff:.8e}")

    full_output = y_elastic_full
    ratios = [0.25, 0.50, 0.75, 1.00]
    rows = []

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase3_sliced_latency.csv"

    for ratio in ratios:
        k = int(elastic.d_inner * ratio)
        y_sliced, debug = elastic(x, active_channels=k, force_prefix_path=True, return_debug=True)
        mse_to_full = torch.mean((y_sliced - full_output) ** 2).item()
        max_abs_diff = (y_sliced - full_output).abs().max().item()
        latency_stats = measure_latency_stats_ms(
            lambda: elastic(x, active_channels=k, force_prefix_path=True),
            warmup=warmup,
            iters=iters,
            device=device,
        )

        print(
            f"ratio={ratio:.2f}, k={k}, mse_to_full={mse_to_full:.8e}, "
            f"max_abs_diff={max_abs_diff:.8e}, mean_latency_ms={latency_stats['mean_latency_ms']:.4f}, "
            f"median_latency_ms={latency_stats['median_latency_ms']:.4f}, p95_latency_ms={latency_stats['p95_latency_ms']:.4f}"
        )
        print(
            "  shapes: "
            f"xz={debug['xz_shape']}, "
            f"conv_input={debug['conv_input_shape']}, "
            f"x_proj_input={debug['x_proj_input_shape']}, "
            f"dt_proj_output={debug['dt_proj_output_shape']}, "
            f"A={debug['A_shape']}, "
            f"out_proj_input={debug['out_proj_input_shape']}"
        )

        rows.append(
            {
                "ratio": f"{ratio:.2f}",
                "active_channels": k,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "d_model": args.d_model,
                "d_inner": elastic.d_inner,
                "mse_to_full": f"{mse_to_full:.8e}",
                "max_abs_diff": f"{max_abs_diff:.8e}",
                "mean_latency_ms": f"{latency_stats['mean_latency_ms']:.6f}",
                "median_latency_ms": f"{latency_stats['median_latency_ms']:.6f}",
                "p95_latency_ms": f"{latency_stats['p95_latency_ms']:.6f}",
                "xz_shape": str(debug["xz_shape"]),
                "conv_input_shape": str(debug["conv_input_shape"]),
                "x_proj_input_shape": str(debug["x_proj_input_shape"]),
                "dt_proj_output_shape": str(debug["dt_proj_output_shape"]),
                "A_shape": str(debug["A_shape"]),
                "out_proj_input_shape": str(debug["out_proj_input_shape"]),
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ratio",
                "active_channels",
                "batch_size",
                "seq_len",
                "d_model",
                "d_inner",
                "mse_to_full",
                "max_abs_diff",
                "mean_latency_ms",
                "median_latency_ms",
                "p95_latency_ms",
                "xz_shape",
                "conv_input_shape",
                "x_proj_input_shape",
                "dt_proj_output_shape",
                "A_shape",
                "out_proj_input_shape",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
