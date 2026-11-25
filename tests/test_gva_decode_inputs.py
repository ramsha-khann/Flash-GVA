# Bad shapes must be rejected before the kernel runs. These checks are on the host, so no GPU is needed.
import pytest
import torch
from flash_gva import gva_decode_attention

B, H, G, T = 2, 8, 2, 100
D_H, D_R = 64, 16


def make_inputs():
    q_tilde = torch.randn(B, H, D_H)
    q_rope = torch.randn(B, H, D_R)
    v_cache = torch.randn(B, T, G, D_H)
    k_rope_cache = torch.randn(B, T, D_R)
    seqlens = torch.full((B,), T, dtype=torch.int32)
    return q_tilde, q_rope, v_cache, k_rope_cache, seqlens


def run(q_tilde, q_rope, v_cache, k_rope_cache, seqlens):
    return gva_decode_attention(q_tilde, q_rope, v_cache, k_rope_cache, seqlens, 0.1, num_splits=1)


def test_one_seqlen_for_two_requests():
    q_tilde, q_rope, v_cache, k_rope_cache, seqlens = make_inputs()
    with pytest.raises(AssertionError):
        run(q_tilde, q_rope, v_cache, k_rope_cache, seqlens[:1])


def test_v_cache_has_fewer_requests():
    q_tilde, q_rope, v_cache, k_rope_cache, seqlens = make_inputs()
    with pytest.raises(AssertionError):
        run(q_tilde, q_rope, v_cache[:1], k_rope_cache[:1], seqlens)


def test_q_rope_has_fewer_heads():
    q_tilde, q_rope, v_cache, k_rope_cache, seqlens = make_inputs()
    with pytest.raises(AssertionError):
        run(q_tilde, q_rope[:, :4], v_cache, k_rope_cache, seqlens)


def test_k_rope_cache_shorter_than_v_cache():
    q_tilde, q_rope, v_cache, k_rope_cache, seqlens = make_inputs()
    with pytest.raises(AssertionError):
        run(q_tilde, q_rope, v_cache, k_rope_cache[:, :50], seqlens)


def test_heads_not_divisible_by_groups():
    q_tilde, q_rope, _, k_rope_cache, seqlens = make_inputs()
    v_cache = torch.randn(B, T, 3, D_H)  # 8 heads cannot be split into 3 groups
    with pytest.raises(AssertionError):
        run(q_tilde, q_rope, v_cache, k_rope_cache, seqlens)


def test_last_dim_not_contiguous():
    q_tilde, q_rope, v_cache, k_rope_cache, seqlens = make_inputs()
    q_tilde = torch.randn(B, H, 2 * D_H)[:, :, ::2]
    with pytest.raises(AssertionError):
        run(q_tilde, q_rope, v_cache, k_rope_cache, seqlens)
