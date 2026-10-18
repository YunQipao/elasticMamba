import copy
import csv
import itertools
from pathlib import Path

import torch
from mamba_ssm import Mamba


@torch.no_grad()
def permute_mamba1_expand_channels_(m: Mamba, perm: torch.Tensor):
    """
    对 Mamba1 block 的 expanded channels 做同步重排。
    """
    d_inner = m.d_inner
    perm = perm.to(m.in_proj.weight.device)

    row_perm = torch.cat([perm, perm + d_inner], dim=0)
    m.in_proj.weight.data = m.in_proj.weight.data[row_perm, :].contiguous()
    if m.in_proj.bias is not None:
        m.in_proj.bias.data = m.in_proj.bias.data[row_perm].contiguous()

    m.conv1d.weight.data = m.conv1d.weight.data[perm, :, :].contiguous()
    if m.conv1d.bias is not None:
        m.conv1d.bias.data = m.conv1d.bias.data[perm].contiguous()

    m.x_proj.weight.data = m.x_proj.weight.data[:, perm].contiguous()
    if getattr(m.x_proj, "bias", None) is not None and m.x_proj.bias is not None:
        pass

    m.dt_proj.weight.data = m.dt_proj.weight.data[perm, :].contiguous()
    if m.dt_proj.bias is not None:
        m.dt_proj.bias.data = m.dt_proj.bias.data[perm].contiguous()

    m.A_log.data = m.A_log.data[perm, :].contiguous()
    m.D.data = m.D.data[perm].contiguous()

    m.out_proj.weight.data = m.out_proj.weight.data[:, perm].contiguous()


@torch.no_grad()
def bad_permute_only_inproj_(m: Mamba, perm: torch.Tensor):
    """
    负例：只重排 in_proj，不同步重排后续模块。
    """
    d_inner = m.d_inner
    perm = perm.to(m.in_proj.weight.device)
    row_perm = torch.cat([perm, perm + d_inner], dim=0)
    m.in_proj.weight.data = m.in_proj.weight.data[row_perm, :].contiguous()
    if m.in_proj.bias is not None:
        m.in_proj.bias.data = m.in_proj.bias.data[row_perm].contiguous()


def build_mamba(d_model: int, d_state: int, expand: int, device: str):
    try:
        module = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=expand,
            use_fast_path=False,
        ).to(device)
    except TypeError:
        module = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=expand,
        ).to(device)
        if hasattr(module, "use_fast_path"):
            module.use_fast_path = False
    module.eval()
    return module


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase1_stress_results.csv"

    seeds = [0, 1, 2, 3, 4]
    d_models = [32, 64, 128]
    d_states = [8, 16, 32]
    expand = 2
    seq_lens = [16, 128, 512]
    batch_sizes = [1, 2, 4]

    rows = []
    total = 0
    passed = 0

    for seed, d_model, d_state, seq_len, batch_size in itertools.product(
        seeds, d_models, d_states, seq_lens, batch_sizes
    ):
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        base = build_mamba(d_model, d_state, expand, device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        y = base(x)

        perm = torch.randperm(base.d_inner, device=device)

        good = copy.deepcopy(base).eval()
        permute_mamba1_expand_channels_(good, perm)
        y_good = good(x)
        good_diff = (y - y_good).abs().max().item()

        bad = copy.deepcopy(base).eval()
        bad_permute_only_inproj_(bad, perm)
        y_bad = bad(x)
        bad_diff = (y - y_bad).abs().max().item()

        good_ok = good_diff < 1e-4
        bad_ok = bad_diff > 1e-5
        ok = good_ok and bad_ok
        total += 1
        passed += int(ok)

        print(
            f"seed={seed} d_model={d_model} d_state={d_state} seq_len={seq_len} batch_size={batch_size} "
            f"good max_abs_diff={good_diff:.8e} bad max_abs_diff={bad_diff:.8e} status={'PASS' if ok else 'FAIL'}"
        )

        rows.append(
            {
                "seed": seed,
                "d_model": d_model,
                "d_state": d_state,
                "expand": expand,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "good_max_abs_diff": f"{good_diff:.8e}",
                "bad_max_abs_diff": f"{bad_diff:.8e}",
                "good_pass": str(good_ok),
                "bad_pass": str(bad_ok),
                "pass": str(ok),
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "d_model",
                "d_state",
                "expand",
                "seq_len",
                "batch_size",
                "good_max_abs_diff",
                "bad_max_abs_diff",
                "good_pass",
                "bad_pass",
                "pass",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    pass_rate = passed / total if total else 0.0
    print(f"Saved results to {csv_path}")
    print(f"Pass rate: {passed}/{total} = {pass_rate:.2%}")

    assert passed == total, "Some configurations failed the permutation checks."


if __name__ == "__main__":
    main()
