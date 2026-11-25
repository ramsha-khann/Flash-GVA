import torch

def gva_decode_attention_ref(q_tilde, q_rope, v_cache, k_rope_cache, cache_seqlens, softmax_scale):
    """Plain PyTorch version of gva_decode_attention, computed in fp32. Same arguments and outputs."""
    B, H, d_h = q_tilde.shape
    G = v_cache.shape[2]
    assert H % G == 0
    dtype = torch.promote_types(q_tilde.dtype, torch.float32)

    out = torch.zeros(B, H, d_h, dtype=dtype, device=q_tilde.device)
    lse = torch.full((B, H), float("-inf"), dtype=dtype, device=q_tilde.device)

    for b in range(B):
        L = int(cache_seqlens[b])
        if L == 0:
            continue
        v = v_cache[b, :L].to(dtype).repeat_interleave(H // G, dim=1)  # [L, H, d_h]
        k_rope = k_rope_cache[b, :L].to(dtype)  # [L, d_r]

        scores = torch.einsum("hd,lhd->hl", q_tilde[b].to(dtype), v)
        scores += q_rope[b].to(dtype) @ k_rope.T
        scores *= softmax_scale

        lse[b] = torch.logsumexp(scores, dim=-1)
        out[b] = torch.einsum("hl,lhd->hd", torch.softmax(scores, dim=-1), v)

    return out, lse
