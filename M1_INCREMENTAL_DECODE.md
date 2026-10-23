# M1: cached incremental Decode

`ElasticMamba2HeadBlock` now exposes a real one-token path:

```python
conv_state, ssm_state = block.allocate_inference_cache(batch_size=1, max_seqlen=256)
output = block.step(token, conv_state, ssm_state, active_head_indices=None)
```

The cache uses the native full-width layout: Conv State is
`[batch, d_ssm + 2 * ngroups * d_state, d_conv - 1]`, and SSM State is
`[batch, nheads, headdim, d_state]`. A fixed subset of heads updates only its
corresponding SSM slices; shared B/C features remain full-width. Conv State is
still updated for the complete projected Conv input so the fixed-width path
does not silently lose shared history.

The `forward(..., inference_params=...)` adapter supports a prefill call at
`seqlen_offset == 0` and one-token calls after the offset is advanced. It does
not implement mid-sequence width changes or state restoration. The supported
reference configuration is `d_mlp=0`, SiLU activation, and `nheads % ngroups ==
0`; RMSNorm additionally requires the selected width to be divisible by
`ngroups`.

## Verification

```bash
ENV=/media/sda1/iic_data/anaconda3/envs/elasticmamba126
PYTHONPATH="$PWD" "$ENV/bin/python" -m pytest -q test_elastic_mamba2_head_block_m1.py
```

The native full-forward versus prefill+step test runs only when CUDA is
available. CPU execution of `mamba_ssm==2.3.2.post1` cannot run its Triton
Mamba2 scan and is therefore skipped rather than replaced with replayed
history.
