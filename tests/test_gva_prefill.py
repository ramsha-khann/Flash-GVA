import math
import pytest
import torch

from gva_prefill_triton import gva_prefill_attention


def reference_attention(q_tilde, q_rope, v, k_rope, scale):
    B, S, H, d_h = q_tilde.shape
    G = v.shape[2]

    # Expand grouped values to query heads.
    v_heads = v.repeat_interleave(H // G, dim=2).float()

    # Content scores.
    scores = torch.einsum(
        "bthd,bjhd->bhtj",
        q_tilde.float(),
        v_heads,
    )

    # Positional scores.
    scores += torch.einsum(
        "bthr,bjr->bhtj",
        q_rope.float(),
        k_rope.float(),
    )

    # Causal attention.
    causal_mask = torch.ones(
        S, S, device=q_tilde.device, dtype=torch.bool
    ).triu(1)

    scores = (scores * scale).masked_fill(
        causal_mask, float("-inf")
    )

    probs = torch.softmax(scores, dim=-1)

    return torch.einsum(
        "bhtj,bjhd->bthd",
        probs,
        v_heads,
    )

@pytest.mark.parametrize("S", [1, 65, 129])
def test_prefill_correctness(S):
    torch.manual_seed(0)

    B, H, G = 1, 4, 2
    d_h, d_r = 64, 16

    dtype = torch.bfloat16
    device = "cuda"

    q_tilde = torch.randn(B, S, H, d_h, device=device, dtype=dtype)
    q_rope = torch.randn(B, S, H, d_r, device=device, dtype=dtype)
    v = torch.randn(B, S, G, d_h, device=device, dtype=dtype)
    k_rope = torch.randn(B, S, d_r, device=device, dtype=dtype)

    scale = 1.0 / math.sqrt(d_h + d_r)

    actual = gva_prefill_attention(
        q_tilde, q_rope, v, k_rope, scale,
        block_m=128,
        block_n=64,
    )

    expected = reference_attention(
        q_tilde, q_rope, v, k_rope, scale
    )

    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=1e-2,
        atol=1e-2,
    )
    