import torch


@torch.no_grad()
def make_head_index(perm: torch.Tensor, headdim: int):
    perm = perm.to(dtype=torch.long)
    offsets = torch.arange(headdim, device=perm.device, dtype=torch.long)
    return (perm[:, None] * headdim + offsets[None, :]).reshape(-1)


@torch.no_grad()
def get_mamba2_layout(mixer):
    total_rows = mixer.in_proj.weight.shape[0]
    d_ssm = int(getattr(mixer, "d_ssm"))
    d_state = int(getattr(mixer, "d_state"))
    ngroups = int(getattr(mixer, "ngroups"))
    nheads = int(getattr(mixer, "nheads"))
    headdim = int(getattr(mixer, "headdim"))
    d_mlp = (total_rows - 2 * d_ssm - 2 * ngroups * d_state - nheads) // 2
    expected_rows = 2 * d_mlp + 2 * d_ssm + 2 * ngroups * d_state + nheads
    if expected_rows != total_rows:
        raise ValueError(
            f"Unexpected in_proj rows: expected {expected_rows} from layout formula, got {total_rows}"
        )
    if d_ssm != nheads * headdim:
        raise ValueError(f"Expected d_ssm == nheads * headdim, got {d_ssm} vs {nheads} * {headdim}")

    z0_start = 0
    x0_start = z0_start + d_mlp
    z_start = x0_start + d_mlp
    x_start = z_start + d_ssm
    b_start = x_start + d_ssm
    c_start = b_start + ngroups * d_state
    dt_start = c_start + ngroups * d_state

    return {
        "d_mlp": d_mlp,
        "d_ssm": d_ssm,
        "d_state": d_state,
        "ngroups": ngroups,
        "nheads": nheads,
        "headdim": headdim,
        "in_proj": {
            "z": (z_start, z_start + d_ssm),
            "x": (x_start, x_start + d_ssm),
            "B": (b_start, b_start + ngroups * d_state),
            "C": (c_start, c_start + ngroups * d_state),
            "dt": (dt_start, dt_start + nheads),
        },
        "conv1d": {
            "x": (0, d_ssm),
            "B": (d_ssm, d_ssm + ngroups * d_state),
            "C": (d_ssm + ngroups * d_state, d_ssm + 2 * ngroups * d_state),
        },
    }


@torch.no_grad()
def _permute_linear_rows_(linear, start: int, end: int, index: torch.Tensor):
    linear.weight.data[start:end, :] = linear.weight.data[start:end, :][index, :].contiguous()
    if linear.bias is not None:
        linear.bias.data[start:end] = linear.bias.data[start:end][index].contiguous()


@torch.no_grad()
def _permute_conv1d_channels_(conv1d, start: int, end: int, index: torch.Tensor):
    conv1d.weight.data[start:end, :, :] = conv1d.weight.data[start:end, :, :][index, :, :].contiguous()
    if conv1d.bias is not None:
        conv1d.bias.data[start:end] = conv1d.bias.data[start:end][index].contiguous()


@torch.no_grad()
def _permute_tensor_1d_(tensor, index: torch.Tensor):
    tensor.data = tensor.data[index].contiguous()


@torch.no_grad()
def _permute_norm_head_block_(norm_module, head_index: torch.Tensor):
    if norm_module is None:
        return
    if hasattr(norm_module, "weight") and isinstance(norm_module.weight, torch.Tensor):
        if norm_module.weight.numel() == head_index.numel():
            norm_module.weight.data = norm_module.weight.data[head_index].contiguous()
    if hasattr(norm_module, "bias") and isinstance(norm_module.bias, torch.Tensor):
        if norm_module.bias.numel() == head_index.numel():
            norm_module.bias.data = norm_module.bias.data[head_index].contiguous()


@torch.no_grad()
def _permute_out_proj_input_(out_proj, head_index: torch.Tensor):
    out_proj.weight.data[:, : head_index.numel()] = out_proj.weight.data[:, : head_index.numel()][:, head_index].contiguous()


@torch.no_grad()
def _permute_in_proj_BC_(mixer, perm: torch.Tensor):
    layout = get_mamba2_layout(mixer)
    ngroups = layout["ngroups"]
    d_state = layout["d_state"]
    nheads = layout["nheads"]
    if ngroups != 1:
        raise ValueError(f"permute_BC ablation currently expects ngroups=1, got {ngroups}")
    if d_state % nheads != 0:
        raise ValueError(f"Cannot split B/C over heads evenly: d_state={d_state}, nheads={nheads}")
    state_per_head = d_state // nheads
    bc_index = make_head_index(perm, state_per_head)
    for block_name in ["B", "C"]:
        start, end = layout["in_proj"][block_name]
        _permute_linear_rows_(mixer.in_proj, start, end, bc_index)
    conv_b_start, conv_b_end = layout["conv1d"]["B"]
    conv_c_start, conv_c_end = layout["conv1d"]["C"]
    _permute_conv1d_channels_(mixer.conv1d, conv_b_start, conv_b_end, bc_index)
    _permute_conv1d_channels_(mixer.conv1d, conv_c_start, conv_c_end, bc_index)


@torch.no_grad()
def permute_mamba2_heads_(mixer, perm: torch.Tensor, ablate=None):
    layout = get_mamba2_layout(mixer)
    nheads = layout["nheads"]
    headdim = layout["headdim"]
    if perm.numel() != nheads:
        raise ValueError(f"perm length {perm.numel()} does not match nheads={nheads}")

    device = mixer.in_proj.weight.device
    perm = perm.to(device=device, dtype=torch.long)
    head_index = make_head_index(perm, headdim)

    z_start, z_end = layout["in_proj"]["z"]
    x_start, x_end = layout["in_proj"]["x"]
    dt_start, dt_end = layout["in_proj"]["dt"]
    conv_x_start, conv_x_end = layout["conv1d"]["x"]

    _permute_linear_rows_(mixer.in_proj, z_start, z_end, head_index)
    _permute_linear_rows_(mixer.in_proj, x_start, x_end, head_index)
    _permute_linear_rows_(mixer.in_proj, dt_start, dt_end, perm)

    if ablate != "no_conv":
        _permute_conv1d_channels_(mixer.conv1d, conv_x_start, conv_x_end, head_index)
    if ablate != "no_A_log":
        _permute_tensor_1d_(mixer.A_log, perm)
    if ablate != "no_D":
        if mixer.D.numel() == nheads:
            _permute_tensor_1d_(mixer.D, perm)
        elif mixer.D.numel() == head_index.numel():
            _permute_tensor_1d_(mixer.D, head_index)
        else:
            raise ValueError(f"Unsupported D shape for permutation: {tuple(mixer.D.shape)}")
    if ablate != "no_dt_bias":
        _permute_tensor_1d_(mixer.dt_bias, perm)
    if ablate != "no_norm":
        _permute_norm_head_block_(getattr(mixer, "norm", None), head_index)
    if ablate != "no_out_proj":
        _permute_out_proj_input_(mixer.out_proj, head_index)
    if ablate == "permute_BC":
        _permute_in_proj_BC_(mixer, perm)


@torch.no_grad()
def bad_permute_mamba2_inproj_only_(mixer, perm: torch.Tensor):
    layout = get_mamba2_layout(mixer)
    device = mixer.in_proj.weight.device
    perm = perm.to(device=device, dtype=torch.long)
    head_index = make_head_index(perm, layout["headdim"])

    z_start, z_end = layout["in_proj"]["z"]
    x_start, x_end = layout["in_proj"]["x"]
    dt_start, dt_end = layout["in_proj"]["dt"]

    _permute_linear_rows_(mixer.in_proj, z_start, z_end, head_index)
    _permute_linear_rows_(mixer.in_proj, x_start, x_end, head_index)
    _permute_linear_rows_(mixer.in_proj, dt_start, dt_end, perm)
