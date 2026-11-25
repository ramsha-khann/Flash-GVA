import pytest
import torch
from flash_gva import gva_prefill_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def prefill_ref(q_tilde, q_rope, v, k_rope, scale):
    S, H, G = q_tilde.shape[1], q_tilde.shape[2], v.shape[2]
    v = v.float().repeat_interleave(H // G, dim=2)
    scores = torch.einsum("bthd,bjhd->bhtj", q_tilde.float(), v)
    scores += torch.einsum("bthr,bjr->bhtj", q_rope.float(), k_rope.float())
    causal = torch.ones(S, S, dtype=torch.bool, device=v.device).tril()
    scores = (scores * scale).masked_fill(~causal, float("-inf"))
    return torch.einsum("bhtj,bjhd->bthd", torch.softmax(scores, dim=-1), v)


@pytest.mark.parametrize("S", [1, 65, 129])
def test_prefill_bf16_matches_reference(S):
    torch.manual_seed(0)
    B, H, G, d_h, d_r = 1, 4, 2, 64, 16
    q_tilde = torch.randn(B, S, H, d_h, device="cuda", dtype=torch.bfloat16)
    q_rope = torch.randn(B, S, H, d_r, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, G, d_h, device="cuda", dtype=torch.bfloat16)
    k_rope = torch.randn(B, S, d_r, device="cuda", dtype=torch.bfloat16)
    scale = (d_h + d_r) ** -0.5

    out = gva_prefill_attention(q_tilde, q_rope, v, k_rope, scale)
    torch.testing.assert_close(out.float(), prefill_ref(q_tilde, q_rope, v, k_rope, scale), rtol=1e-2, atol=1e-2)
