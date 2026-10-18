import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.modules import mamba_simple
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops import selective_scan_interface


class ElasticMamba1Block(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.default_active_channels = None
        self.default_force_prefix_path = False
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @classmethod
    def from_mamba(cls, mamba: Mamba):
        block = cls(
            d_model=mamba.d_model,
            d_state=mamba.d_state,
            d_conv=mamba.d_conv,
            expand=mamba.expand,
            dt_rank=mamba.dt_rank,
            conv_bias=mamba.conv1d.bias is not None,
            bias=mamba.in_proj.bias is not None,
            layer_idx=mamba.layer_idx,
            device=mamba.in_proj.weight.device,
            dtype=mamba.in_proj.weight.dtype,
        )
        block.load_state_dict(mamba.state_dict(), strict=True)
        return block

    def _validate_active_channels(self, active_channels):
        if active_channels is None:
            return self.d_inner
        active_channels = int(active_channels)
        if active_channels < 1 or active_channels > self.d_inner:
            raise ValueError(f"active_channels must be in [1, {self.d_inner}], got {active_channels}")
        return active_channels

    def set_active_channels(self, active_channels, force_prefix_path: bool = False):
        self.default_active_channels = active_channels
        self.default_force_prefix_path = force_prefix_path

    def _forward_full(self, hidden_states):
        _, seqlen, _ = hidden_states.shape
        xz = rearrange(self.in_proj(hidden_states), "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        x = mamba_simple.causal_conv1d_fn(
            x=x,
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=self.activation,
        ) if mamba_simple.causal_conv1d_fn is not None else self.act(self.conv1d(x)[..., :seqlen])
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        A = -torch.exp(self.A_log.float())
        y = mamba_simple.selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)

    def _forward_prefix(self, hidden_states, active_channels, return_debug: bool = False):
        _, seqlen, _ = hidden_states.shape
        k = self._validate_active_channels(active_channels)

        x = F.linear(
            hidden_states,
            self.in_proj.weight[:k, :],
            self.in_proj.bias[:k] if self.in_proj.bias is not None else None,
        )
        z = F.linear(
            hidden_states,
            self.in_proj.weight[self.d_inner : self.d_inner + k, :],
            self.in_proj.bias[self.d_inner : self.d_inner + k] if self.in_proj.bias is not None else None,
        )
        x = rearrange(x, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l")
        xz = torch.cat([x, z], dim=1)
        conv_input = x

        conv_bias = self.conv1d.bias[:k] if self.conv1d.bias is not None else None
        can_use_causal_conv1d = (
            mamba_simple.causal_conv1d_fn is not None
            and x.is_cuda
            and k % 8 == 0
        )
        if can_use_causal_conv1d:
            x = mamba_simple.causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight[:k], "d 1 w -> d w"),
                bias=conv_bias,
                activation=self.activation,
            )
        else:
            x = F.conv1d(
                x,
                self.conv1d.weight[:k, :, :],
                conv_bias,
                padding=self.d_conv - 1,
                groups=k,
            )
            x = self.act(x)[..., :seqlen]

        x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), self.x_proj.weight[:, :k], None)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight[:k, :], None)
        dt = rearrange(dt, "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        A = -torch.exp(self.A_log[:k].float())
        can_use_fast_selective_scan = (
            mamba_simple.selective_scan_fn is not None
            and x.is_cuda
            and k % 8 == 0
        )
        selective_scan_impl = (
            mamba_simple.selective_scan_fn
            if can_use_fast_selective_scan
            else selective_scan_interface.selective_scan_ref
        )
        y = selective_scan_impl(
            x,
            dt,
            A,
            B,
            C,
            self.D[:k].float(),
            z=z,
            delta_bias=self.dt_proj.bias[:k].float(),
            delta_softplus=True,
        )
        y = rearrange(y, "b d l -> b l d")
        out_proj_input = y
        out = F.linear(y, self.out_proj.weight[:, :k], self.out_proj.bias)

        if not return_debug:
            return out

        debug = {
            "active_channels": k,
            "xz_shape": tuple(xz.shape),
            "conv_input_shape": tuple(conv_input.shape),
            "x_proj_input_shape": tuple(rearrange(x, "b d l -> (b l) d").shape),
            "dt_proj_output_shape": tuple(dt.shape),
            "A_shape": tuple(A.shape),
            "out_proj_input_shape": tuple(out_proj_input.shape),
        }
        return out, debug

    def forward(self, hidden_states, active_channels=None, inference_params=None, force_prefix_path=False, return_debug=False, **kwargs):
        if active_channels is None:
            active_channels = self.default_active_channels
            force_prefix_path = force_prefix_path or self.default_force_prefix_path
        if active_channels is None or int(active_channels) == self.d_inner:
            if force_prefix_path:
                return self._forward_prefix(hidden_states, self.d_inner, return_debug=return_debug)
            return self._forward_full(hidden_states)
        return self._forward_prefix(hidden_states, active_channels, return_debug=return_debug)
