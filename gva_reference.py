import torch

def gva_decode_attention_ref(
    q_tilde: torch.Tensor,       # [B, H, d_h]
    q_rope: torch.Tensor,        # [B, H, d_r]
    v_cache: torch.Tensor,       # [B, T, G, d_h]
    k_rope_cache: torch.Tensor,  # [B, T, d_r]
    cache_seqlens: torch.Tensor, # [B]
    softmax_scale: float,
):
    """
    PyTorch reference for GVA decode attention.
    Returns:
        output: [B, H, d_h]
        lse:    [B, H]
    """

    B, H, d_h = q_tilde.shape
    G = v_cache.shape[2]

    assert H % G == 0

    compute_dtype = torch.promote_types(
        q_tilde.dtype, torch.float32
    )

    output = torch.zeros(
        B, H, d_h,
        device=q_tilde.device,
        dtype=compute_dtype,
    )

    lse = torch.full(
        (B, H),
        float("-inf"),
        device=q_tilde.device,
        dtype=compute_dtype,
    )

    for b in range(B):
        L = int(cache_seqlens[b])

        if L == 0:
            continue

        # Select valid cached values.
        v = v_cache[b, :L].to(compute_dtype)
        k_rope = k_rope_cache[b, :L].to(compute_dtype)

        # Expand grouped values to query heads.
        v = v.repeat_interleave(H // G, dim=1)

        q_content = q_tilde[b].to(compute_dtype)
        q_pos = q_rope[b].to(compute_dtype)

        # Content scores + positional scores.
        scores = torch.einsum(
            "hd,lhd->hl", q_content, v
        )

        scores += q_pos @ k_rope.T

        scores *= softmax_scale

        # Softmax and output.
        lse[b] = torch.logsumexp(scores, dim=-1)

        probs = torch.softmax(scores, dim=-1)

        output[b] = torch.einsum(
            "hl,lhd->hd", probs, v
        )

    return output, lse