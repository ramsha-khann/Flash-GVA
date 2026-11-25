import pytest
import torch
from flash_gva import gva_decode_attention
from flash_gva.reference import gva_decode_attention_ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

B, T, H, G, D_H, D_R = 3, 200, 8, 2, 64, 16
SCALE = (D_H + D_R) ** -0.5


def values(shape, start):
    n = torch.arange(torch.Size(shape).numel()) + start
    return ((n % 7 - 3) / 4).reshape(shape).to("cuda", torch.bfloat16)


@pytest.mark.parametrize("num_splits", [1, 3, None])
def test_decode_cuda_graph(num_splits):
    # Capture once, then change the lengths and cache contents in place and replay.
    q_tilde = values((B, H, D_H), start=0)
    q_rope = values((B, H, D_R), start=1)
    v_cache = values((B, T, G, D_H), start=2)
    k_rope_cache = values((B, T, D_R), start=3)
    seqlens = torch.full((B,), T, dtype=torch.int32, device="cuda")
    args = (q_tilde, q_rope, v_cache, k_rope_cache, seqlens, SCALE)

    gva_decode_attention(*args, num_splits=num_splits)  # compile before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = gva_decode_attention(*args, num_splits=num_splits)

    for step, lengths in enumerate([[1, 1, 1], [0, 64, 200], [200, 7, 130]]):
        q_tilde.copy_(values((B, H, D_H), start=10 + step))
        q_rope.copy_(values((B, H, D_R), start=20 + step))
        v_cache.copy_(values((B, T, G, D_H), start=30 + step))
        k_rope_cache.copy_(values((B, T, D_R), start=40 + step))
        for b, L in enumerate(lengths):
            v_cache[b, L:] = float("nan")  # unused rows must be ignored
            k_rope_cache[b, L:] = float("nan")
        seqlens.copy_(torch.tensor(lengths, dtype=torch.int32))
        graph.replay()

        out_ref, lse_ref = gva_decode_attention_ref(*args)
        torch.testing.assert_close(out.float(), out_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse, lse_ref, atol=2e-2, rtol=2e-2)
