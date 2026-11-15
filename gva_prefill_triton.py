import math
import torch
import triton
import triton.language as tl

@triton.jit
def _attend_tiles(
    acc, m_i, l_i, q_tilde, q_rope, v_base, kr_base,
    stride_v_s, stride_kr_s, offs_m, start, end, seqlen, qk_scale,
    D_H: tl.constexpr, D_R: tl.constexpr, BLOCK_DR: tl.constexpr, BLOCK_N: tl.constexpr,
    ON_DIAGONAL: tl.constexpr,
):

    offs_dh = tl.arange(0, D_H)
    offs_dr = tl.arange(0, BLOCK_DR)
    for start_n in range(start, end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        in_prompt = offs_n < seqlen  # the last tile may run past the end of the prompt
        v = tl.load(
            v_base + offs_n[:, None] * stride_v_s + offs_dh[None, :],
            mask=in_prompt[:, None], other=0.0,
        )  # [BLOCK_N, D_H]
        k_rope = tl.load(
            kr_base + offs_n[:, None] * stride_kr_s + offs_dr[None, :],
            mask=in_prompt[:, None] & (offs_dr[None, :] < D_R), other=0.0,
        )  # [BLOCK_N, BLOCK_DR]

        s = tl.dot(q_tilde, tl.trans(v)) + tl.dot(q_rope, tl.trans(k_rope))  # [BLOCK_M, BLOCK_N]
        s = s * qk_scale  # includes log2(e), so exp2 below is exp
        if ON_DIAGONAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))  # token t sees j <= t

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)  # rescales what was summed before this tile
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)  # the same V tile, now as values
        m_i = m_new
    return acc, m_i, l_i

@triton.jit
def _gva_prefill_kernel(
    Q_tilde, Q_rope, V, K_rope, O,
    seqlen, qk_scale,
    stride_qt_b, stride_qt_s, stride_qt_h,
    stride_qr_b, stride_qr_s, stride_qr_h,
    stride_v_b, stride_v_s, stride_v_g,
    stride_kr_b, stride_kr_s,
    stride_o_b, stride_o_s, stride_o_h,
    NUM_HEADS: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D_H: tl.constexpr,
    D_R: tl.constexpr,
    BLOCK_DR: tl.constexpr,
    BLOCK_M: tl.constexpr,  # query tokens per program
    BLOCK_N: tl.constexpr,  # key tokens per loop step
):
    bh = tl.program_id(0)
    b = (bh // NUM_HEADS).to(tl.int64)
    h = bh % NUM_HEADS
    g = h // HEADS_PER_GROUP  # value group of head h
    start_m = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dh = tl.arange(0, D_H)
    offs_dr = tl.arange(0, BLOCK_DR)
    q_in_prompt = offs_m < seqlen

    q_tilde = tl.load(
        Q_tilde + b * stride_qt_b + h * stride_qt_h + offs_m[:, None] * stride_qt_s + offs_dh[None, :],
        mask=q_in_prompt[:, None], other=0.0,
    )  # [BLOCK_M, D_H]
    q_rope = tl.load(
        Q_rope + b * stride_qr_b + h * stride_qr_h + offs_m[:, None] * stride_qr_s + offs_dr[None, :],
        mask=q_in_prompt[:, None] & (offs_dr[None, :] < D_R), other=0.0,
    )  # [BLOCK_M, BLOCK_DR]
    v_base = V + b * stride_v_b + g * stride_v_g
    kr_base = K_rope + b * stride_kr_b

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # running max of each row
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # running softmax denominator
    acc = tl.zeros([BLOCK_M, D_H], dtype=tl.float32)  # running output

    diag_start = start_m * BLOCK_M
    acc, m_i, l_i = _attend_tiles(
        acc, m_i, l_i, q_tilde, q_rope, v_base, kr_base, stride_v_s, stride_kr_s,
        offs_m, 0, diag_start, seqlen, qk_scale,
        D_H, D_R, BLOCK_DR, BLOCK_N, ON_DIAGONAL=False,
    )
    diag_end = tl.minimum(diag_start + BLOCK_M, seqlen)
    acc, m_i, l_i = _attend_tiles(
        acc, m_i, l_i, q_tilde, q_rope, v_base, kr_base, stride_v_s, stride_kr_s,
        offs_m, diag_start, diag_end, seqlen, qk_scale,
        D_H, D_R, BLOCK_DR, BLOCK_N, ON_DIAGONAL=True,
    )

    o = acc / l_i[:, None]
    tl.store(
        O + b * stride_o_b + h * stride_o_h + offs_m[:, None] * stride_o_s + offs_dh[None, :],
        o.to(O.dtype.element_ty),
        mask=q_in_prompt[:, None],
    )


def gva_prefill_attention(
    q_tilde, q_rope, v, k_rope, softmax_scale,
    block_m=128, block_n=64, num_warps=8, num_stages=3,
):
    B, S, H, d_h = q_tilde.shape
    G, d_r = v.shape[2], q_rope.shape[-1]
    assert q_rope.shape == (B, S, H, d_r) and v.shape == (B, S, G, d_h) and k_rope.shape == (B, S, d_r)
    assert H % G == 0
    assert q_tilde.dtype in (torch.bfloat16, torch.float16)
    assert q_rope.dtype == v.dtype == k_rope.dtype == q_tilde.dtype
    assert d_h >= 16 and d_h & (d_h - 1) == 0
    assert block_m % block_n == 0, "the diagonal must start on a key-tile boundary"
    for t in (q_tilde, q_rope, v, k_rope):
        assert t.stride(-1) == 1, "the kernel indexes the last dim with stride 1"
    assert block_m >= 16
    assert block_n >= 16

    assert (block_m & (block_m - 1)) == 0
    assert (block_n & (block_n - 1)) == 0

    assert block_m % block_n == 0
    
    o = torch.empty(B, S, H, d_h, dtype=q_tilde.dtype, device=q_tilde.device)
    grid = (B * H, triton.cdiv(S, block_m))
    _gva_prefill_kernel[grid](
        q_tilde, q_rope, v, k_rope, o,
        S, softmax_scale * math.log2(math.e),
        q_tilde.stride(0), q_tilde.stride(1), q_tilde.stride(2),
        q_rope.stride(0), q_rope.stride(1), q_rope.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        k_rope.stride(0), k_rope.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        NUM_HEADS=H,
        HEADS_PER_GROUP=H // G,
        D_H=d_h,
        D_R=d_r,
        BLOCK_DR=max(16, triton.next_power_of_2(d_r)),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o
