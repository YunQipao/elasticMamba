import copy
import torch
from mamba_ssm import Mamba


@torch.no_grad()
def permute_mamba1_expand_channels_(m: Mamba, perm: torch.Tensor):
    """
    对 Mamba1 block 的 expanded channels 做同步重排。

    假设：
    - in_proj 输出 2 * d_inner，前半是 x branch，后半是 z branch；
    - conv1d 是 depthwise conv，按 d_inner 通道排列；
    - x_proj 的输入维度是 d_inner，因此需要重排列；
    - dt_proj 的输出维度是 d_inner，因此需要重排行；
    - A_log / D 按 d_inner 排列；
    - out_proj 的输入维度是 d_inner，因此需要重排列。
    """
    d_inner = m.d_inner
    perm = perm.to(m.in_proj.weight.device)

    # 1. in_proj: rows of x branch and z branch
    row_perm = torch.cat([perm, perm + d_inner], dim=0)
    m.in_proj.weight.data = m.in_proj.weight.data[row_perm, :].contiguous()
    if m.in_proj.bias is not None:
        m.in_proj.bias.data = m.in_proj.bias.data[row_perm].contiguous()

    # 2. depthwise conv1d: channel dimension
    m.conv1d.weight.data = m.conv1d.weight.data[perm, :, :].contiguous()
    if m.conv1d.bias is not None:
        m.conv1d.bias.data = m.conv1d.bias.data[perm].contiguous()

    # 3. x_proj: input channel dimension, i.e., columns
    m.x_proj.weight.data = m.x_proj.weight.data[:, perm].contiguous()
    if getattr(m.x_proj, "bias", None) is not None and m.x_proj.bias is not None:
        # x_proj 通常没有 bias；有 bias 时不需要重排，因为它属于输出维度
        pass

    # 4. dt_proj: output channel dimension, i.e., rows
    m.dt_proj.weight.data = m.dt_proj.weight.data[perm, :].contiguous()
    if m.dt_proj.bias is not None:
        m.dt_proj.bias.data = m.dt_proj.bias.data[perm].contiguous()

    # 5. SSM parameters
    m.A_log.data = m.A_log.data[perm, :].contiguous()
    m.D.data = m.D.data[perm].contiguous()

    # 6. out_proj: input channel dimension, i.e., columns
    m.out_proj.weight.data = m.out_proj.weight.data[:, perm].contiguous()
    # out_proj.bias belongs to output hidden dimension, no permutation needed


@torch.no_grad()
def bad_permute_only_inproj_(m: Mamba, perm: torch.Tensor):
    """
    负例：只重排 in_proj，不同步重排后续模块。
    输出应该明显变化，用来证明“必须同步重排整条路径”。
    """
    d_inner = m.d_inner
    perm = perm.to(m.in_proj.weight.device)
    row_perm = torch.cat([perm, perm + d_inner], dim=0)
    m.in_proj.weight.data = m.in_proj.weight.data[row_perm, :].contiguous()
    if m.in_proj.bias is not None:
        m.in_proj.bias.data = m.in_proj.bias.data[row_perm].contiguous()


def main():
    torch.manual_seed(0)
    device = "cuda"

    batch, seqlen, d_model = 2, 128, 64

    # 为了更容易 debug，先关闭 fast path。
    try:
        base = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            use_fast_path=False,
        ).to(device)
    except TypeError:
        base = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to(device)
        if hasattr(base, "use_fast_path"):
            base.use_fast_path = False

    base.eval()
    x = torch.randn(batch, seqlen, d_model, device=device)

    y = base(x)

    perm = torch.randperm(base.d_inner, device=device)

    # 正例：同步重排整条 expanded channel path
    good = copy.deepcopy(base).eval()
    permute_mamba1_expand_channels_(good, perm)
    y_good = good(x)
    good_diff = (y - y_good).abs().max().item()

    # 负例：只重排 in_proj
    bad = copy.deepcopy(base).eval()
    bad_permute_only_inproj_(bad, perm)
    y_bad = bad(x)
    bad_diff = (y - y_bad).abs().max().item()

    print(f"d_inner = {base.d_inner}")
    print(f"good max_abs_diff = {good_diff:.8e}")
    print(f"bad  max_abs_diff = {bad_diff:.8e}")

    assert good_diff < 1e-4, "同步重排后输出不等价，说明维度映射有错误。"
    assert bad_diff > 1e-5, "负例差异过小，可能测试没有覆盖到重排路径。"

    print("[PASS] Mamba1 expanded-channel permutation equivalence verified.")


if __name__ == "__main__":
    main()
