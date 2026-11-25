"""
Same inputs, three ways:
  gqa:           plain GQA with a G-head key cache
  reconstructed: GVA as written in the paper, k = [v @ M_h, k_rope]
  absorbed:      our kernels, q_tilde = q_nope @ M_h^T (paper eq. 7)
"""
import os

import pytest
import torch
import torch.nn.functional as F
from flash_gva import gva_decode_attention, gva_prefill_attention

DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
if DEVICE == "cuda" and not torch.cuda.is_available():
    pytest.skip("needs a GPU or TRITON_INTERPRET=1", allow_module_level=True)

B, S, H, G = 1, 5, 4, 2  # head h uses value group h // 2
D_H, D_N, D_R = 16, 8, 4
SCALE = (D_N + D_R) ** -0.5


def values(shape, start):
    # -0.75 .. 0.75 in steps of 0.25, exact in fp16
    n = torch.arange(torch.Size(shape).numel()) + start
    return ((n % 7 - 3) / 4).reshape(shape).to(DEVICE)


q_nope = values((B, S, H, D_N), start=0)
q_rope = values((B, S, H, D_R), start=1)
v = values((B, S, G, D_H), start=2)
k_rope = values((B, S, D_R), start=3)
M = values((H, D_H, D_N), start=4)  # one key map per head


def gqa(k):
    q = torch.cat([q_nope, q_rope], dim=-1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, scale=SCALE, enable_gqa=True,
    )
    return out.transpose(1, 2)


def reconstructed(M):
    v_h = v.repeat_interleave(H // G, dim=2)
    k_nope = torch.einsum("bshd,hdn->bshn", v_h, M)  # paper eq. 6
    k = torch.cat([k_nope, k_rope[:, :, None, :].expand(B, S, H, D_R)], dim=-1)
    q = torch.cat([q_nope, q_rope], dim=-1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v_h.transpose(1, 2), is_causal=True, scale=SCALE,
    )
    return out.transpose(1, 2)


def absorbed_prefill(M):
    q_tilde = torch.einsum("bshn,hdn->bshd", q_nope, M)
    return gva_prefill_attention(q_tilde.half(), q_rope.half(), v.half(), k_rope.half(), SCALE).float()


def absorbed_decode(M):
    # last token against the whole cache
    q_tilde = torch.einsum("bhn,hdn->bhd", q_nope[:, -1], M)
    seqlens = torch.full((B,), S, dtype=torch.int32, device=DEVICE)
    out, _ = gva_decode_attention(
        q_tilde.half(), q_rope[:, -1].half(), v.half(), k_rope.half(), seqlens, SCALE, num_splits=1,
    )
    return out.float()


def test_prefill():
    torch.testing.assert_close(absorbed_prefill(M), reconstructed(M), atol=1e-2, rtol=1e-2)


def test_decode():
    torch.testing.assert_close(absorbed_decode(M), reconstructed(M)[:, -1], atol=1e-2, rtol=1e-2)


def test_matches_gqa_when_heads_share_m():
    # With one M per group, the heads of a group have the same keys, so they fit in a G-head GQA cache.
    M_group = values((G, D_H, D_N), start=4)
    M_shared = M_group.repeat_interleave(H // G, dim=0)
    k_nope = torch.einsum("bsgd,gdn->bsgn", v, M_group)
    k = torch.cat([k_nope, k_rope[:, :, None, :].expand(B, S, G, D_R)], dim=-1)
    expected = gqa(k)

    torch.testing.assert_close(reconstructed(M_shared), expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(absorbed_prefill(M_shared), expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(absorbed_decode(M_shared), expected[:, -1], atol=1e-2, rtol=1e-2)


def test_differs_from_gqa_with_unrelated_keys():
    # Checks that the comparison can fail: keys that are not v @ M give a different result.
    k = values((B, S, G, D_N + D_R), start=5)
    assert not torch.allclose(absorbed_prefill(M), gqa(k), atol=1e-2, rtol=1e-2)
