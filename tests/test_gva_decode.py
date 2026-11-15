"""
Run the file:  pytest -v -s tests/test_gva_decode.py
Without a GPU, run it in Triton's CPU interpreter:  TRITON_INTERPRET=1 pytest -v -s tests/test_gva_decode.py
"""
import os

import pytest
import torch
from gva_decode_triton import gva_decode_attention
from gva_reference import gva_decode_attention_ref

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"


@pytest.mark.skipif(
    not INTERPRET and not torch.cuda.is_available(),
    reason="needs a CUDA GPU, or TRITON_INTERPRET=1 to run on CPU",
)
def test_gva_decode_matches_reference():
    torch.manual_seed(0)
    batch = 1
    heads = 8
    groups = 4
    seqlen = 65
    head_dim = 32
    rope_dim = 16

    device = DEVICE
    dtype = torch.float32

    q_tilde = torch.randn(
        batch, heads, head_dim, device=device, dtype=dtype
    )
    q_rope = torch.randn(
        batch, heads, rope_dim, device=device, dtype=dtype
    )

    v_cache = torch.randn(
        batch, seqlen, groups, head_dim,
        device=device, dtype=dtype,
    )
    k_rope_cache = torch.randn(
        batch, seqlen, rope_dim, device=device, dtype=dtype
    )

    cache_seqlens = torch.tensor(
        [seqlen], device=device, dtype=torch.int32
    )

    scale = 1.0 / (head_dim + rope_dim) ** 0.5

    expected, expected_lse = gva_decode_attention_ref(
        q_tilde,
        q_rope,
        v_cache,
        k_rope_cache,
        cache_seqlens,
        scale,
    )

    actual, actual_lse = gva_decode_attention(
        q_tilde,
        q_rope,
        v_cache,
        k_rope_cache,
        cache_seqlens,
        scale,
        num_splits=1,
    )

    torch.testing.assert_close(
        actual, expected, rtol=1e-3, atol=1e-3
    )

    torch.testing.assert_close(
        actual_lse, expected_lse, rtol=1e-3, atol=1e-3
    )