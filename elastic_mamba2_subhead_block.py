import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from mamba2_permutation_utils import get_mamba2_layout


class ElasticMamba2SubheadBlock(nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.d_model = mixer.d_model
        self.d_state = mixer.d_state
        self.d_conv = mixer.d_conv
        self.expand = mixer.expand
        self.d_inner = mixer.d_inner
        self.d_ssm = mixer.d_ssm
        self.headdim = mixer.headdim
        self.nheads = mixer.nheads
        self.ngroups = mixer.ngroups
        self.D_has_hdim = mixer.D_has_hdim
        self.rmsnorm = mixer.rmsnorm
        self.norm_before_gate = mixer.norm_before_gate
        self.chunk_size = mixer.chunk_size
        self.activation = mixer.activation
        self.layer_idx = getattr(mixer, "layer_idx", None)

        self.in_proj = copy.deepcopy(mixer.in_proj)
        self.conv1d = copy.deepcopy(mixer.conv1d)
        self.act = nn.SiLU()
        self.dt_bias = nn.Parameter(mixer.dt_bias.detach().clone())
        self.A_log = nn.Parameter(mixer.A_log.detach().clone())
        self.D = nn.Parameter(mixer.D.detach().clone())
        self.norm = copy.deepcopy(getattr(mixer, "norm", None)) if self.rmsnorm else None
        self.out_proj = copy.deepcopy(mixer.out_proj)

        self.layout = get_mamba2_layout(mixer)
        if self.layout["d_mlp"] != 0:
            raise ValueError(
                f"ElasticMamba2SubheadBlock currently supports d_mlp=0 only, got {self.layout['d_mlp']}"
            )
        self.default_active_head_indices = None
        self.default_per_head_active_channel_indices = None

    @classmethod
    def from_mamba2(cls, mixer):
        return cls(mixer)

    def set_active_head_indices(self, active_head_indices):
        self.default_active_head_indices = active_head_indices

    def set_per_head_active_channel_indices(self, per_head_active_channel_indices):
        self.default_per_head_active_channel_indices = per_head_active_channel_indices

    def _resolve_active_head_indices(self, active_head_indices, device):
        if active_head_indices is None:
            active_head_indices = self.default_active_head_indices
        if active_head_indices is None:
            return torch.arange(self.nheads, device=device, dtype=torch.long)
        head_indices = torch.as_tensor(active_head_indices, device=device, dtype=torch.long)
        if head_indices.numel() < 1 or head_indices.numel() > self.nheads:
            raise ValueError(f"active_head_indices must select 1..{self.nheads} heads, got {head_indices.numel()}")
        if head_indices.min().item() < 0 or head_indices.max().item() >= self.nheads:
            raise ValueError(f"active_head_indices out of range [0, {self.nheads - 1}]")
        if torch.unique(head_indices).numel() != head_indices.numel():
            raise ValueError("active_head_indices must not contain duplicates")
        return head_indices

    def _resolve_per_head_active_channel_indices(self, active_heads, per_head_active_channel_indices, device):
        if per_head_active_channel_indices is None:
            per_head_active_channel_indices = self.default_per_head_active_channel_indices

        selected = []
        if per_head_active_channel_indices is None:
            for head_idx in active_heads.tolist():
                selected.append(torch.arange(self.headdim, device=device, dtype=torch.long) + head_idx * self.headdim)
            return selected

        for head_idx in active_heads.tolist():
            if head_idx not in per_head_active_channel_indices:
                raise ValueError(f"Missing channel selection for active head {head_idx}")
            channel_indices = torch.as_tensor(
                per_head_active_channel_indices[head_idx],
                device=device,
                dtype=torch.long,
            )
            if channel_indices.numel() < 1 or channel_indices.numel() > self.headdim:
                raise ValueError(
                    f"Head {head_idx} must select 1..{self.headdim} channels, got {channel_indices.numel()}"
                )
            if channel_indices.min().item() < 0 or channel_indices.max().item() >= self.headdim:
                raise ValueError(f"Head {head_idx} channel indices out of range [0, {self.headdim - 1}]")
            if torch.unique(channel_indices).numel() != channel_indices.numel():
                raise ValueError(f"Head {head_idx} channel indices contain duplicates")
            channel_indices = torch.sort(channel_indices).values
            selected.append(channel_indices + head_idx * self.headdim)
        return selected

    def forward(
        self,
        u,
        seqlen=None,
        seq_idx=None,
        cu_seqlens=None,
        inference_params=None,
        active_head_indices=None,
        per_head_active_channel_indices=None,
        **kwargs,
    ):
        if inference_params is not None:
            raise NotImplementedError("ElasticMamba2SubheadBlock does not support inference cache path")
        if seqlen is not None or seq_idx is not None or cu_seqlens is not None:
            raise NotImplementedError("ElasticMamba2SubheadBlock currently supports standard batched forward only")

        _, seqlen, _ = u.shape
        device = u.device
        active_heads = self._resolve_active_head_indices(active_head_indices, device)
        per_head_channel_indices = self._resolve_per_head_active_channel_indices(
            active_heads,
            per_head_active_channel_indices,
            device,
        )
        selected_index = torch.cat(per_head_channel_indices, dim=0)
        per_head_widths = [indices.numel() for indices in per_head_channel_indices]

        zxbcdt = self.in_proj(u)
        z_start, z_end = self.layout["in_proj"]["z"]
        x_start, x_end = self.layout["in_proj"]["x"]
        b_start, b_end = self.layout["in_proj"]["B"]
        c_start, c_end = self.layout["in_proj"]["C"]
        dt_start, dt_end = self.layout["in_proj"]["dt"]

        z = zxbcdt[..., z_start:z_end][..., selected_index]
        x = zxbcdt[..., x_start:x_end][..., selected_index]
        B = zxbcdt[..., b_start:b_end]
        C = zxbcdt[..., c_start:c_end]
        dt = zxbcdt[..., dt_start:dt_end][..., active_heads]

        conv_b_start, conv_b_end = self.layout["conv1d"]["B"]
        conv_c_start, conv_c_end = self.layout["conv1d"]["C"]
        conv_index = torch.cat(
            [
                selected_index,
                torch.arange(conv_b_start, conv_b_end, device=device, dtype=torch.long),
                torch.arange(conv_c_start, conv_c_end, device=device, dtype=torch.long),
            ],
        )
        xBC = torch.cat([x, B, C], dim=-1)
        xBC = F.conv1d(
            xBC.transpose(1, 2),
            self.conv1d.weight[conv_index, :, :],
            self.conv1d.bias[conv_index] if self.conv1d.bias is not None else None,
            padding=self.d_conv - 1,
            groups=conv_index.numel(),
        ).transpose(1, 2)[:, :seqlen]
        xBC = self.act(xBC)

        selected_dim = selected_index.numel()
        x = xBC[..., :selected_dim]
        B = xBC[..., selected_dim : selected_dim + (b_end - b_start)]
        C = xBC[..., selected_dim + (b_end - b_start) :]

        A = -torch.exp(self.A_log[active_heads].float())
        dt_bias = self.dt_bias[active_heads]
        if self.D.numel() == self.nheads:
            D = self.D[active_heads]
        elif self.D.numel() == self.d_ssm:
            flat_D = self.D[selected_index]
            D = [flat_D[offset : offset + width] for offset, width in zip(_prefix_offsets(per_head_widths), per_head_widths)]
        else:
            raise ValueError(f"Unsupported D shape: {tuple(self.D.shape)}")

        x_heads = _split_flat_features(x, per_head_widths)
        z_heads = _split_flat_features(z, per_head_widths)
        if isinstance(D, list):
            D_heads = D
        else:
            D_heads = [D[idx] for idx in range(D.shape[0])]

        y_heads = []
        for head_offset, width in enumerate(per_head_widths):
            x_head = x_heads[head_offset].unsqueeze(2).contiguous()
            z_head = z_heads[head_offset].unsqueeze(2).contiguous()
            if D_heads[head_offset].ndim == 0:
                D_head = D_heads[head_offset].reshape(1)
            elif D_heads[head_offset].ndim == 1:
                D_head = D_heads[head_offset].reshape(1, width)
            else:
                D_head = D_heads[head_offset]
            y_head = mamba_chunk_scan_combined(
                x_head,
                dt[..., head_offset : head_offset + 1],
                A[head_offset : head_offset + 1],
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=D_head,
                z=None if self.rmsnorm else z_head,
                dt_bias=dt_bias[head_offset : head_offset + 1],
                dt_softplus=True,
                return_final_states=False,
            )
            y_heads.append(y_head.squeeze(2))
        y = torch.cat(y_heads, dim=-1)

        if self.rmsnorm:
            y = _apply_per_head_rmsnorm(
                y,
                z,
                self.norm,
                selected_index,
                per_head_widths,
                self.ngroups,
                self.norm_before_gate,
            )

        return F.linear(y, self.out_proj.weight[:, selected_index], self.out_proj.bias)


def _prefix_offsets(widths):
    offsets = []
    running = 0
    for width in widths:
        offsets.append(running)
        running += width
    return offsets


def _split_flat_features(tensor, widths):
    outputs = []
    offset = 0
    for width in widths:
        outputs.append(tensor[..., offset : offset + width])
        offset += width
    return outputs


def _apply_per_head_rmsnorm(y, z, norm, selected_index, per_head_widths, ngroups: int, norm_before_gate: bool):
    weight = norm.weight[selected_index]
    bias = norm.bias[selected_index] if getattr(norm, "bias", None) is not None else None
    selected_dim = selected_index.numel()
    if selected_dim % ngroups != 0:
        raise ValueError(
            f"Selected dim {selected_dim} must be divisible by ngroups={ngroups} for RMSNorm"
        )
    return rms_norm_ref(
        y,
        weight,
        bias,
        z=z,
        eps=norm.eps,
        group_size=selected_dim // ngroups,
        norm_before_gate=norm_before_gate,
        upcast=True,
    )
