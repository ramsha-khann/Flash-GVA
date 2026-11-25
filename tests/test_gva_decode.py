import os

import pytest
import torch
from flash_gva import gva_decode_attention
from flash_gva.reference import gva_decode_attention_ref

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"


@pytest.mark.skipif(not INTERPRET and not torch.cuda.is_available(), reason="needs a GPU or TRITON_INTERPRET=1")
def test_decode_fp32_matches_reference():
    torch.manual_seed(0)
    B, H, G, L, d_h, d_r = 1, 8, 4, 65, 32, 16
    q_tilde = torch.randn(B, H, d_h, device=DEVICE)
    q_rope = torch.randn(B, H, d_r, device=DEVICE)
    v_cache = torch.randn(B, L, G, d_h, device=DEVICE)
    k_rope_cache = torch.randn(B, L, d_r, device=DEVICE)
    seqlens = torch.tensor([L], dtype=torch.int32, device=DEVICE)
    scale = (d_h + d_r) ** -0.5

    out, lse = gva_decode_attention(q_tilde, q_rope, v_cache, k_rope_cache, seqlens, scale, num_splits=1)
    out_ref, lse_ref = gva_decode_attention_ref(q_tilde, q_rope, v_cache, k_rope_cache, seqlens, scale)

    torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)
