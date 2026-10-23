"""M1 tests for the explicit cached ElasticMamba2HeadBlock path.

The native Mamba2 full-forward comparison tests require a CUDA device because
mamba_ssm 2.3.2.post1 routes its reference scan through Triton.  Those tests
are skipped on CPU rather than replaced by a history-replay approximation.
"""

import pytest
import torch

from elastic_mamba2_head_block import ElasticMamba2HeadBlock


def make_block(*, ngroups=4, rmsnorm=True):
    from mamba_ssm.modules.mamba2 import Mamba2

    mixer = Mamba2(
        d_model=256,
        d_state=64,
        ngroups=ngroups,
        headdim=32,
        rmsnorm=rmsnorm,
        device="cpu",
        dtype=torch.float32,
    )
    mixer.layer_idx = 0
    return ElasticMamba2HeadBlock(mixer).eval()


def run_steps(block, x, heads=None):
    conv, ssm = block.allocate_inference_cache(x.shape[0], x.shape[1], device=x.device)
    outputs = []
    for token in x.split(1, dim=1):
        outputs.append(block.step(token, conv, ssm, heads))
    return torch.cat(outputs, dim=1), conv, ssm


def test_full_width_step_is_finite_and_updates_both_states():
    torch.manual_seed(0)
    block = make_block()
    x = torch.randn(1, 32, block.d_model)
    conv0, ssm0 = block.allocate_inference_cache(1, 32)
    out, conv, ssm = run_steps(block, x)

    assert out.shape == (1, 32, block.d_model)
    assert torch.isfinite(out).all()
    assert conv.shape == (1, block.d_ssm + 2 * block.ngroups * block.d_state, block.d_conv - 1)
    assert ssm.shape == (1, block.nheads, block.headdim, block.d_state)
    assert not torch.equal(conv, conv0)
    assert not torch.equal(ssm, ssm0)


@pytest.mark.parametrize("count", [8, 12])
@pytest.mark.parametrize("length", [32, 128, 256])
def test_fixed_head_width_step_is_finite(count, length):
    torch.manual_seed(1)
    block = make_block()
    x = torch.randn(1, length, block.d_model)
    heads = torch.arange(count)
    out, conv, ssm = run_steps(block, x, heads)
    assert out.shape == (1, length, block.d_model)
    assert torch.isfinite(out).all()
    assert torch.isfinite(conv).all() and torch.isfinite(ssm).all()
    assert torch.count_nonzero(ssm[:, count:]) == 0


@pytest.mark.parametrize(
    "heads, message",
    [([-1], "out of range"), ([16], "out of range"), ([1, 1], "duplicates")],
)
def test_invalid_head_indices_raise(heads, message):
    block = make_block()
    conv, ssm = block.allocate_inference_cache(1, 1)
    with pytest.raises(ValueError, match=message):
        block.step(torch.zeros(1, 1, block.d_model), conv, ssm, heads)


def test_unsupported_mlp_configuration_is_explicit():
    block = make_block()
    block.layout["d_mlp"] = 1
    conv, ssm = block.allocate_inference_cache(1, 1)
    with pytest.raises(ValueError, match="d_mlp=0"):
        block.step(torch.zeros(1, 1, block.d_model), conv, ssm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba2 Triton reference path requires CUDA")
def test_cuda_native_prefill_and_step_alignment():
    """Compare native Mamba2 full forward with real cached token stepping."""
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.utils.generation import InferenceParams

    torch.manual_seed(2)
    mixer = Mamba2(
        d_model=256, d_state=64, ngroups=4, headdim=32,
        rmsnorm=True, device="cuda", dtype=torch.float32,
    ).eval()
    mixer.layer_idx = 0
    block = ElasticMamba2HeadBlock(mixer).eval()
    prefix_len = 48
    continuation_len = 16
    x = torch.randn(1, prefix_len + continuation_len, 256, device="cuda")
    reference = mixer(x)
    params = InferenceParams(max_seqlen=x.shape[1], max_batch_size=1)
    block(x[:, :prefix_len], inference_params=params)
    params.seqlen_offset = prefix_len
    actual = []
    for index in range(prefix_len, prefix_len + continuation_len):
        actual.append(block(x[:, index:index + 1], inference_params=params))
        params.seqlen_offset += 1
    actual = torch.cat(actual, dim=1)
    torch.cuda.synchronize()
    assert torch.allclose(actual, reference[:, prefix_len:], atol=2e-4, rtol=2e-4)
