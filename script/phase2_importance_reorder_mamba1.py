import copy
import torch
from mamba_ssm import Mamba
from pathlib import Path

from phase1_check_mamba1_permutation import permute_mamba1_expand_channels_


def add_score(score, tensor, dim, index):
    """
    对某个 tensor 的某个维度上的 index 切片计算 |grad * weight|。
    """
    if tensor.grad is None:
        return score

    if dim == 0:
        part = (tensor.grad[index] * tensor[index]).detach().abs().sum()
    elif dim == 1:
        part = (tensor.grad[:, index] * tensor[:, index]).detach().abs().sum()
    else:
        raise ValueError("Only dim=0 or dim=1 is supported.")
    return score + part


def compute_mamba1_channel_importance(m: Mamba):
    """
    对 Mamba1 的每个 expanded channel 计算 importance(U_i)=sum |grad * weight|。

    这里把一个 channel path 视为：
    - in_proj 的 x row 和 z row
    - conv1d 的 channel
    - x_proj 的输入列
    - dt_proj 的输出行
    - A_log 的行
    - D 的元素
    - out_proj 的输入列
    """
    d_inner = m.d_inner
    device = m.in_proj.weight.device
    scores = torch.zeros(d_inner, device=device)

    with torch.no_grad():
        for i in range(d_inner):
            s = torch.tensor(0.0, device=device)

            # in_proj rows: x branch i, z branch i
            s = add_score(s, m.in_proj.weight, dim=0, index=i)
            s = add_score(s, m.in_proj.weight, dim=0, index=i + d_inner)
            if m.in_proj.bias is not None and m.in_proj.bias.grad is not None:
                s += (m.in_proj.bias.grad[i] * m.in_proj.bias[i]).abs()
                s += (m.in_proj.bias.grad[i + d_inner] * m.in_proj.bias[i + d_inner]).abs()

            # conv1d channel
            s = add_score(s, m.conv1d.weight, dim=0, index=i)
            if m.conv1d.bias is not None and m.conv1d.bias.grad is not None:
                s += (m.conv1d.bias.grad[i] * m.conv1d.bias[i]).abs()

            # x_proj input column
            s = add_score(s, m.x_proj.weight, dim=1, index=i)

            # dt_proj output row
            s = add_score(s, m.dt_proj.weight, dim=0, index=i)
            if m.dt_proj.bias is not None and m.dt_proj.bias.grad is not None:
                s += (m.dt_proj.bias.grad[i] * m.dt_proj.bias[i]).abs()

            # A_log row
            s = add_score(s, m.A_log, dim=0, index=i)

            # D element
            if m.D.grad is not None:
                s += (m.D.grad[i] * m.D[i]).abs()

            # out_proj input column
            s = add_score(s, m.out_proj.weight, dim=1, index=i)

            scores[i] = s

    return scores


def main():
    torch.manual_seed(1)
    device = "cuda"

    batch, seqlen, d_model = 4, 128, 64

    try:
        model = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            use_fast_path=False,
        ).to(device)
    except TypeError:
        model = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to(device)
        if hasattr(model, "use_fast_path"):
            model.use_fast_path = False

    model.train()

    x = torch.randn(batch, seqlen, d_model, device=device)

    # 这里只是机制验证，所以用 toy loss。
    # 后面迁移到 LM 时要换成 next-token prediction loss。
    y = model(x)
    loss = y.pow(2).mean()

    model.zero_grad(set_to_none=True)
    loss.backward()

    scores = compute_mamba1_channel_importance(model)
    perm = torch.argsort(scores, descending=True)

    print("top-10 scores:", scores[perm[:10]].detach().cpu().tolist())
    print("top-10 indices:", perm[:10].detach().cpu().tolist())

    original = copy.deepcopy(model).eval()
    reordered = copy.deepcopy(model).eval()
    permute_mamba1_expand_channels_(reordered, perm)

    with torch.no_grad():
        y0 = original(x)
        y1 = reordered(x)
        diff = (y0 - y1).abs().max().item()

    print(f"reordered full-model max_abs_diff = {diff:.8e}")
    assert diff < 1e-4, "按 importance 重排后，完整模型输出不等价。"

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "scores": scores.detach().cpu(),
            "perm": perm.detach().cpu(),
        },
        out_dir / "mamba1_channel_importance.pt",
    )

    print("[PASS] One-shot importance scoring and reordering verified.")


if __name__ == "__main__":
    main()
