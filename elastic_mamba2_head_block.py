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

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, device=None, **kwargs):
        """Allocate the same full-width cache layout used by Mamba2.

        ``max_seqlen`` is accepted for API compatibility; recurrent state size
        is independent of it.  The cache is ``(conv_state, ssm_state)`` with
        shapes ``[B, d_inner + 2 * ngroups * d_state, d_conv - 1]`` and
        ``[B, nheads, headdim, d_state]``.
        """
        if device is None:
            device = self.in_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        conv_width = self.d_ssm + 2 * self.ngroups * self.d_state
        conv_state = torch.zeros(
            batch_size, conv_width, max(self.d_conv - 1, 0), device=device, dtype=conv_dtype
        )
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

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

    def _validate_runtime_config(self, active_heads):
        if self.layout["d_mlp"] != 0:
            raise ValueError("incremental decode supports d_mlp=0 only")
        if self.activation not in ("silu", "swish"):
            raise ValueError(f"incremental decode supports silu activation only, got {self.activation!r}")
        if self.nheads % self.ngroups != 0:
            raise ValueError(
                f"incremental decode requires nheads divisible by ngroups, got {self.nheads} and {self.ngroups}"
            )
        if self.rmsnorm and (active_heads.numel() * self.headdim) % self.ngroups != 0:
            raise ValueError(
                "RMSNorm incremental decode requires selected head width divisible by ngroups; "
                f"got {active_heads.numel() * self.headdim} and ngroups={self.ngroups}"
            )

    def _project(self, u, active_heads):
        zxbcdt = self.in_proj(u)
        z_start, z_end = self.layout["in_proj"]["z"]
        x_start, x_end = self.layout["in_proj"]["x"]
        b_start, b_end = self.layout["in_proj"]["B"]
        c_start, c_end = self.layout["in_proj"]["C"]
        dt_start, dt_end = self.layout["in_proj"]["dt"]
        head_index = make_head_index(active_heads, self.headdim)
        return (
            zxbcdt[..., z_start:z_end][..., head_index],
            zxbcdt[..., x_start:x_end],
            zxbcdt[..., b_start:b_end],
            zxbcdt[..., c_start:c_end],
            zxbcdt[..., dt_start:dt_end][..., active_heads],
            head_index,
        )

    def _causal_conv(self, xbc, active_heads, conv_state=None):
        head_index = make_head_index(active_heads, self.headdim)
        conv_width = xbc.shape[-1]
        conv_input = xbc.transpose(1, 2)
        if conv_state is None:
            weights = self.conv1d.weight
            bias = self.conv1d.bias
            output = F.conv1d(conv_input, weights, bias, padding=self.d_conv - 1, groups=conv_width)
            return self.act(output.transpose(1, 2)[:, : xbc.shape[1]]), xbc

        if conv_state.ndim != 3 or conv_state.shape[1] != conv_width or conv_state.shape[2] != max(self.d_conv - 1, 0):
            raise ValueError(
                "conv_state must have shape "
                f"[batch, {conv_width}, {max(self.d_conv - 1, 0)}], got {tuple(conv_state.shape)}"
            )
        pieces = []
        for token in range(xbc.shape[1]):
            window = torch.cat([conv_state, xbc[:, token : token + 1, :].transpose(1, 2)], dim=-1)
            current = F.conv1d(window, self.conv1d.weight, self.conv1d.bias, groups=conv_width)
            pieces.append(current.transpose(1, 2))
            if conv_state.shape[-1] > 0:
                conv_state.copy_(torch.cat([conv_state[..., 1:], xbc[:, token : token + 1, :].transpose(1, 2)], dim=-1))
        return self.act(torch.cat(pieces, dim=1)), xbc

    def _scan(self, x, z, B, C, dt, active_heads, return_final_states=False):
        A = -torch.exp(self.A_log[active_heads].float())
        if self.D.numel() == self.nheads:
            D = self.D[active_heads]
        elif self.D.numel() == self.d_ssm:
            D = rearrange(self.D[make_head_index(active_heads, self.headdim)], "(h p) -> h p", p=self.headdim)
        else:
            raise ValueError(f"Unsupported D shape: {tuple(self.D.shape)}")
        return mamba_chunk_scan_combined(
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
            return_final_states=return_final_states,
        )

    def _finish(self, y, z, head_index):
        if self.rmsnorm:
            y = rms_norm_ref(
                y,
                self.norm.weight[head_index],
                self.norm.bias[head_index] if getattr(self.norm, "bias", None) is not None else None,
                z=z,
                eps=self.norm.eps,
                group_size=head_index.numel() // self.ngroups,
                norm_before_gate=self.norm_before_gate,
                upcast=True,
            )
        return F.linear(y, self.out_proj.weight[:, head_index], self.out_proj.bias)

    def _cache_for(self, inference_params, batch_size, dtype, device):
        if self.layer_idx is None:
            raise ValueError("inference_params requires mixer.layer_idx to identify the cache entry")
        cache_dict = getattr(inference_params, "key_value_memory_dict", None)
        if cache_dict is None:
            raise TypeError("inference_params must expose key_value_memory_dict")
        if self.layer_idx not in cache_dict:
            cache_dict[self.layer_idx] = self.allocate_inference_cache(
                batch_size, inference_params.max_seqlen, dtype=dtype, device=device
            )
        cache = cache_dict[self.layer_idx]
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise TypeError("ElasticMamba2HeadBlock cache must be (conv_state, ssm_state)")
        return cache

    @torch.no_grad()
    def step(self, u, conv_state, ssm_state, active_head_indices=None):
        """Run one token and mutate the supplied full-width states in place."""
        if u.ndim == 2:
            u = u.unsqueeze(1)
        if u.ndim != 3 or u.shape[1] != 1:
            raise ValueError(f"step expects [batch, 1, d_model] or [batch, d_model], got {tuple(u.shape)}")
        active_heads = self._resolve_active_head_indices(active_head_indices, u.device)
        self._validate_runtime_config(active_heads)
        z, x_full, B_full, C_full, dt, head_index = self._project(u, active_heads)
        xbc = torch.cat([x_full, B_full, C_full], dim=-1)
        conv_out, _ = self._causal_conv(xbc, active_heads, conv_state)
        selected = head_index.numel()
        x = conv_out[..., : self.d_ssm][..., head_index]
        B = conv_out[..., self.d_ssm : self.d_ssm + self.ngroups * self.d_state]
        C = conv_out[..., self.d_ssm + self.ngroups * self.d_state :]
        dt = F.softplus(dt + self.dt_bias[active_heads])
        A = -torch.exp(self.A_log[active_heads].float())
        group_heads = self.nheads // self.ngroups
        B = B.reshape(B.shape[0], 1, self.ngroups, self.d_state).repeat_interleave(group_heads, dim=2)
        C = C.reshape(C.shape[0], 1, self.ngroups, self.d_state).repeat_interleave(group_heads, dim=2)
        B = B[:, :, active_heads, :]
        C = C[:, :, active_heads, :]
        xh = x.reshape(x.shape[0], 1, active_heads.numel(), self.headdim)
        d_a = torch.exp(dt[:, 0, :].float()[:, :, None, None] * A[None, :, None, None])
        selected_state = ssm_state[:, active_heads, :, :].clone()
        selected_state.mul_(d_a).add_(
            xh[:, 0].float()[..., None]
            * dt[:, 0, :].float()[:, :, None, None]
            * B[:, 0].float()[:, :, None, :]
        )
        ssm_state[:, active_heads, :, :] = selected_state
        y = selected_state.mul(C[:, 0].float()[:, :, None, :]).sum(dim=-1)
        if self.D.numel() == self.nheads:
            y = y + xh[:, 0].float() * self.D[active_heads][None, :, None]
        elif self.D.numel() == self.d_ssm:
            y = y + xh[:, 0].float() * self.D[head_index].reshape(1, active_heads.numel(), self.headdim)
        else:
            raise ValueError(f"Unsupported D shape: {tuple(self.D.shape)}")
        y = self._finish(y.reshape(y.shape[0], 1, selected), z, head_index)
        return y

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
            if seqlen is not None or seq_idx is not None or cu_seqlens is not None:
                raise NotImplementedError("cached path does not support seqlen/seq_idx/cu_seqlens")
            if u.ndim != 3:
                raise ValueError(f"cached forward expects [batch, seqlen, d_model], got {tuple(u.shape)}")
            offset = int(getattr(inference_params, "seqlen_offset", 0))
            conv_state, ssm_state = self._cache_for(inference_params, u.shape[0], u.dtype, u.device)
            active_heads = self._resolve_active_head_indices(active_head_indices, u.device)
            if offset > 0:
                if u.shape[1] != 1:
                    raise ValueError("cached decode accepts exactly one token after prefill")
                return self.step(u, conv_state, ssm_state, active_heads)
            if u.shape[1] == 1:
                return self.step(u, conv_state, ssm_state, active_heads)
            self._validate_runtime_config(active_heads)
            z, x_full, B_full, C_full, dt, head_index = self._project(u, active_heads)
            conv_out, raw_xbc = self._causal_conv(torch.cat([x_full, B_full, C_full], dim=-1), active_heads)
            conv_state.copy_(raw_xbc[:, -conv_state.shape[-1] :, :].transpose(1, 2) if conv_state.shape[-1] else conv_state)
            selected = head_index.numel()
            x = conv_out[..., : self.d_ssm][..., head_index]
            B = conv_out[..., self.d_ssm : self.d_ssm + self.ngroups * self.d_state]
            C = conv_out[..., self.d_ssm + self.ngroups * self.d_state :]
            scan_result = self._scan(x, z, B, C, dt, active_heads, return_final_states=True)
            y, final_state = scan_result
            ssm_state[:, active_heads].copy_(final_state)
            return self._finish(rearrange(y, "b l h p -> b l (h p)"), z, head_index)
        if seqlen is not None or seq_idx is not None or cu_seqlens is not None:
            raise NotImplementedError("ElasticMamba2HeadBlock currently supports standard batched forward only")

        batch, seqlen, _ = u.shape
        device = u.device
        active_heads = self._resolve_active_head_indices(active_head_indices, device)
        self._validate_runtime_config(active_heads)
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
