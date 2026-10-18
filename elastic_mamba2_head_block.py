import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from mamba2_permutation_utils import get_mamba2_layout, make_head_index


class ElasticMamba2HeadBlock(nn.Module):
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
                f"ElasticMamba2HeadBlock currently supports d_mlp=0 only, got {self.layout['d_mlp']}"
            )
        self.default_active_head_indices = None

    @classmethod
    def from_mamba2(cls, mixer):
        return cls(mixer)

    def set_active_head_indices(self, active_head_indices):
        self.default_active_head_indices = active_head_indices

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
        return head_indices

    def forward(
        self,
        u,
        seqlen=None,
        seq_idx=None,
        cu_seqlens=None,
        inference_params=None,
        active_head_indices=None,
        **kwargs,
    ):
        if inference_params is not None:
            raise NotImplementedError("ElasticMamba2HeadBlock does not support inference cache path")
        if seqlen is not None or seq_idx is not None or cu_seqlens is not None:
            raise NotImplementedError("ElasticMamba2HeadBlock currently supports standard batched forward only")

        batch, seqlen, _ = u.shape
        device = u.device
        active_heads = self._resolve_active_head_indices(active_head_indices, device)
        head_index = make_head_index(active_heads, self.headdim)

        zxbcdt = self.in_proj(u)
        z_start, z_end = self.layout["in_proj"]["z"]
        x_start, x_end = self.layout["in_proj"]["x"]
        b_start, b_end = self.layout["in_proj"]["B"]
        c_start, c_end = self.layout["in_proj"]["C"]
        dt_start, dt_end = self.layout["in_proj"]["dt"]

        z = zxbcdt[..., z_start:z_end][..., head_index]
        x = zxbcdt[..., x_start:x_end][..., head_index]
        B = zxbcdt[..., b_start:b_end]
        C = zxbcdt[..., c_start:c_end]
        dt = zxbcdt[..., dt_start:dt_end][..., active_heads]

        conv_b_start, conv_b_end = self.layout["conv1d"]["B"]
        conv_c_start, conv_c_end = self.layout["conv1d"]["C"]
        conv_index = torch.cat(
            [
                head_index,
                torch.arange(conv_b_start, conv_b_end, device=device, dtype=torch.long),
                torch.arange(conv_c_start, conv_c_end, device=device, dtype=torch.long),
            ],
            dim=0,
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

        selected_dim = head_index.numel()
        x = xBC[..., :selected_dim]
        B = xBC[..., selected_dim : selected_dim + (b_end - b_start)]
        C = xBC[..., selected_dim + (b_end - b_start) :]

        A = -torch.exp(self.A_log[active_heads].float())
        if self.D.numel() == self.nheads:
            D = self.D[active_heads]
        elif self.D.numel() == self.d_ssm:
            D = rearrange(self.D[head_index], "(h p) -> h p", p=self.headdim)
        else:
            raise ValueError(f"Unsupported D shape: {tuple(self.D.shape)}")

        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=D,
            z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
            dt_bias=self.dt_bias[active_heads],
            dt_softplus=True,
            return_final_states=False,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        if self.rmsnorm:
            y = rms_norm_ref(
                y,
                self.norm.weight[head_index],
                self.norm.bias[head_index] if getattr(self.norm, "bias", None) is not None else None,
                z=z,
                eps=self.norm.eps,
                group_size=selected_dim // self.ngroups,
                norm_before_gate=self.norm_before_gate,
                upcast=True,
            )

        return F.linear(y, self.out_proj.weight[:, head_index], self.out_proj.bias)
