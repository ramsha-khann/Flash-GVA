import math
import torch
import triton
import triton.language as tl
from triton.runtime import driver

@triton.jit
def _gva_decode_split_kernel(
    Q_tilde, Q_rope, V, K_rope, Seqlens, O_part, LSE_part,
    qk_scale, num_splits,
    stride_qt_b, stride_qt_h,
    stride_qr_b, stride_qr_h,
    stride_v_b, stride_v_t, stride_v_g,
    stride_kr_b, stride_kr_t,
    stride_op_b, stride_op_h, stride_op_s,
    stride_lp_b, stride_lp_h,
    HEADS_PER_GROUP: tl.constexpr,
    D_H: tl.constexpr,
    D_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DH: tl.constexpr,
    BLOCK_DR: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    split = tl.program_id(2)
    b64 = b.to(tl.int64)
    seqlen = tl.load(Seqlens + b64).to(tl.int32)

    split_len = tl.cdiv(tl.cdiv(seqlen, num_splits), BLOCK_N) * BLOCK_N
    split_start = split * split_len
    split_end = tl.minimum(split_start + split_len, seqlen)

    offs_h = g * HEADS_PER_GROUP + tl.arange(0, BLOCK_H)
    mask_h = tl.arange(0, BLOCK_H) < HEADS_PER_GROUP
    offs_dh = tl.arange(0, BLOCK_DH)
    offs_dr = tl.arange(0, BLOCK_DR)
    offs_n = tl.arange(0, BLOCK_N)
    mask_dh = offs_dh < D_H
    mask_dr = offs_dr < D_R

    q_tilde = tl.load(
        Q_tilde + b64 * stride_qt_b + offs_h[:, None] * stride_qt_h + offs_dh[None, :],
        mask=mask_h[:, None] & mask_dh[None, :],
        other=0.0,
    )  # [BLOCK_H, BLOCK_DH]
    q_rope = tl.load(
        Q_rope + b64 * stride_qr_b + offs_h[:, None] * stride_qr_h + offs_dr[None, :],
        mask=mask_h[:, None] & mask_dr[None, :],
        other=0.0,
    )  # [BLOCK_H, BLOCK_DR]

    e_max = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_DH, BLOCK_H], dtype=tl.float32)  # o transposed

    for start in range(split_start, split_end, BLOCK_N):
        n = start + offs_n
        mask_n = n < split_end

        v = tl.load(
            V + b64 * stride_v_b + g * stride_v_g + n[:, None] * stride_v_t + offs_dh[None, :],
            mask=mask_n[:, None] & mask_dh[None, :],
            other=0.0,
        )
        k_rope = tl.load(
            K_rope + b64 * stride_kr_b + n[:, None] * stride_kr_t + offs_dr[None, :],
            mask=mask_n[:, None] & mask_dr[None, :],
            other=0.0,
        )
        s = tl.dot(v, tl.trans(q_tilde)) + tl.dot(k_rope, tl.trans(q_rope))
        s = s * qk_scale
        s = tl.where(mask_n[:, None], s, float("-inf"))

        n_e_max = tl.maximum(tl.max(s, 0), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(s - n_e_max[None, :])
        # o^T += V^T p^T: [BLOCK_DH, BLOCK_N] x [BLOCK_N, BLOCK_H].
        acc = acc * re_scale[None, :] + tl.dot(tl.trans(v), p.to(v.dtype))
        e_sum = e_sum * re_scale + tl.sum(p, 0)
        e_max = n_e_max

    has_tokens = e_sum > 0
    o = acc / tl.where(has_tokens, e_sum, 1.0)[None, :]
    lse = tl.where(has_tokens, (e_max + tl.log2(e_sum)) * 0.6931471805599453, float("-inf"))

    tl.store(
        O_part + b64 * stride_op_b + offs_h[None, :] * stride_op_h + split * stride_op_s
        + offs_dh[:, None],
        o,
        mask=mask_dh[:, None] & mask_h[None, :],
    )
    tl.store(LSE_part + b64 * stride_lp_b + offs_h * stride_lp_h + split, lse, mask=mask_h)


@triton.jit
def _gva_merge_kernel(
    O_part, LSE_part, O, LSE,
    num_splits,
    stride_op_b, stride_op_h, stride_op_s,
    stride_lp_b, stride_lp_h,
    stride_o_b, stride_o_h,
    stride_lse_b,
    D_H: tl.constexpr,
    BLOCK_DH: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)

    offs_s = tl.arange(0, BLOCK_S)
    offs_dh = tl.arange(0, BLOCK_DH)
    mask_s = offs_s < num_splits
    mask_dh = offs_dh < D_H

    lse_s = tl.load(
        LSE_part + b * stride_lp_b + h * stride_lp_h + offs_s, mask=mask_s, other=float("-inf")
    )  # [BLOCK_S]
    o_s = tl.load(
        O_part + b * stride_op_b + h * stride_op_h + offs_s[:, None] * stride_op_s
        + offs_dh[None, :],
        mask=mask_s[:, None] & mask_dh[None, :],
        other=0.0,
    )  # [BLOCK_S, BLOCK_DH]

    lse_max = tl.max(lse_s, 0)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    w = tl.exp(lse_s - lse_max)
    w_sum = tl.sum(w, 0)
    has_tokens = w_sum > 0
    o = tl.sum(w[:, None] * o_s, 0) / tl.where(has_tokens, w_sum, 1.0)
    lse = tl.where(has_tokens, lse_max + tl.log(w_sum), float("-inf"))

    tl.store(O + b * stride_o_b + h * stride_o_h + offs_dh, o, mask=mask_dh)
    tl.store(LSE + b * stride_lse_b + h, lse)


_CTAS_PER_SM = {}  # config -> CTAs of the split kernel that fit on one SM


def _ctas_per_sm(kernel, device_index):
    """CTAs of a compiled kernel that fit on one SM, as in Triton's fused-softmax tutorial."""
    kernel._init_handles()
    props = driver.active.utils.get_device_properties(device_index)
    threads = 32 * kernel.metadata.num_warps
    by_regs = props["max_num_regs"] // (kernel.n_regs * threads)
    by_smem = props["max_shared_mem"] // max(kernel.metadata.shared, 1)
    return max(1, min(by_regs, by_smem, 2048 // threads))


def _num_splits(ctas_per_split, slots, max_splits):
    """Split count that fills the GPU most evenly.

    The GPU runs `slots` CTAs at once (SMs * CTAs per SM). With s splits the grid
    is ctas_per_split * s CTAs; if that is not close to a whole number of full
    waves, some SMs get fewer CTAs and sit idle at the end. Measured on H100
    (B=8, 64k): 256 CTAs = 1.94 per SM took 191 us, 288 CTAs = 2.18 per SM 207 us.
    Take the fewest splits within 5% of the best fill: fewer splits, less merge work.
    """
    # Even the most splits fit in one wave: every extra CTA keeps another SM busy.
    if ctas_per_split * max_splits <= slots:
        return max_splits

    def fill(s):
        waves = ctas_per_split * s / slots
        return waves / math.ceil(waves)

    best = max(fill(s) for s in range(1, max_splits + 1))
    return next(s for s in range(1, max_splits + 1) if fill(s) >= 0.95 * best)


def gva_decode_attention(
    q_tilde, q_rope, v_cache, k_rope_cache, cache_seqlens, softmax_scale,
    num_splits=None, block_n=64, num_stages=3,
):
    B, H, d_h = q_tilde.shape
    T, G = v_cache.shape[1], v_cache.shape[2]
    d_r = q_rope.shape[-1]
    assert H % G == 0 and v_cache.shape[-1] == d_h and k_rope_cache.shape[-1] == d_r
    # tl.dot needs both operands in the same dtype.
    assert q_tilde.dtype == v_cache.dtype and q_rope.dtype == k_rope_cache.dtype
    # Kernel indexes the last dim with stride 1.
    for t in (q_tilde, q_rope, v_cache, k_rope_cache):
        assert t.stride(-1) == 1
    # The kernel reads the lengths from GPU memory. A CPU tensor would need a
    # host->device copy here, which is not allowed during CUDA-graph capture.
    assert cache_seqlens.is_cuda
    # Rows of the transposed dots; wgmma needs >= 64.
    assert block_n >= 64 and block_n & (block_n - 1) == 0

    dev = q_tilde.device
    heads_per_group = H // G
    BLOCK_DH = max(16, triton.next_power_of_2(d_h))
    BLOCK_DR = max(16, triton.next_power_of_2(d_r))
    meta = dict(
        HEADS_PER_GROUP=heads_per_group,
        D_H=d_h,
        D_R=d_r,
        BLOCK_H=max(16, triton.next_power_of_2(heads_per_group)),
        BLOCK_DH=BLOCK_DH,
        BLOCK_DR=BLOCK_DR,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=num_stages,
    )

    def split_args(o_part, lse_part, num_splits):
        return (
            q_tilde, q_rope, v_cache, k_rope_cache, cache_seqlens, o_part, lse_part,
            softmax_scale * math.log2(math.e), num_splits,
            q_tilde.stride(0), q_tilde.stride(1),
            q_rope.stride(0), q_rope.stride(1),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            k_rope_cache.stride(0), k_rope_cache.stride(1),
            o_part.stride(0), o_part.stride(1), o_part.stride(2),
            lse_part.stride(0), lse_part.stride(1),
        )

    if num_splits is None:
        # How many CTAs fit on one SM depends on the compiled kernel (registers,
        # shared memory), so compile once per config to read it. This is host-only
        # and cached, but must happen before CUDA-graph capture, like Triton's
        # first compile.
        key = (dev, q_tilde.dtype, q_rope.dtype, heads_per_group, d_h, d_r, block_n, num_stages)
        if key not in _CTAS_PER_SM:
            o_tmp = torch.empty(B, H, 1, d_h, dtype=torch.float32, device=dev)
            lse_tmp = torch.empty(B, H, 1, dtype=torch.float32, device=dev)
            kernel = _gva_decode_split_kernel.warmup(*split_args(o_tmp, lse_tmp, 1), grid=(B, G, 1), **meta)
            _CTAS_PER_SM[key] = _ctas_per_sm(kernel, dev.index)
        num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
        # Depends only on B, G, T and the GPU, never on cache_seqlens: CUDA-graph safe.
        max_splits = min(triton.cdiv(T, block_n), 64)
        num_splits = _num_splits(B * G, num_sms * _CTAS_PER_SM[key], max_splits)
    num_splits = max(1, num_splits)

    o_part = torch.empty(B, H, num_splits, d_h, dtype=torch.float32, device=dev)
    lse_part = torch.empty(B, H, num_splits, dtype=torch.float32, device=dev)
    _gva_decode_split_kernel[(B, G, num_splits)](*split_args(o_part, lse_part, num_splits), **meta)

    if num_splits == 1:
        return o_part[:, :, 0], lse_part[:, :, 0]

    o = torch.empty(B, H, d_h, dtype=torch.float32, device=dev)
    lse = torch.empty(B, H, dtype=torch.float32, device=dev)
    _gva_merge_kernel[(B, H)](
        o_part, lse_part, o, lse,
        num_splits,
        o_part.stride(0), o_part.stride(1), o_part.stride(2),
        lse_part.stride(0), lse_part.stride(1),
        o.stride(0), o.stride(1),
        lse.stride(0),
        D_H=d_h,
        BLOCK_DH=BLOCK_DH,
        BLOCK_S=triton.next_power_of_2(num_splits),
    )
    return o, lse


def _random_inputs(cfg, B, T, dtype, dev):
    q_tilde = torch.randn(B, cfg.num_heads, cfg.d_h, dtype=dtype, device=dev)
    q_rope = torch.randn(B, cfg.num_heads, cfg.d_r, dtype=dtype, device=dev)
    v_cache = torch.randn(B, T, cfg.num_groups, cfg.d_h, dtype=dtype, device=dev)
    k_rope_cache = torch.randn(B, T, cfg.d_r, dtype=dtype, device=dev)
    return q_tilde, q_rope, v_cache, k_rope_cache


def _check(name, o, lse, inputs, seq_lens, scale, atol):
    from gva_reference import gva_decode_attention_ref, report_diff

    o_ref, lse_ref = gva_decode_attention_ref(*inputs, seq_lens, scale)
    live = seq_lens > 0
    ok = report_diff(f"{name} o", o, o_ref, atol)
    # lse is -inf for empty requests, and -inf - -inf = NaN, so compare those separately.
    ok &= report_diff(f"{name} lse", lse[live], lse_ref[live], atol)
    empty_ok = bool(torch.isneginf(lse[~live]).all())
    print(f"[{'OK' if empty_ok else 'FAIL'}] {name}: lse == -inf for empty requests")
    return ok and empty_ok


def _test():
    from gva_reference import GVAConfig

    torch.manual_seed(0)
    dev = "cuda"
    for dtype, atol in [(torch.float32, 1e-2), (torch.bfloat16, 2e-2)]:
        cfg = GVAConfig(d_model=64, num_heads=16, num_groups=4, d_h=128, d_n=128, d_r=16)
        seq_lens = torch.tensor([0, 1, 63, 64, 65, 1000], device=dev)
        B, T = len(seq_lens), int(seq_lens.max())
        inputs = _random_inputs(cfg, B, T, dtype, dev)
        # Unused cache slots may hold anything, NaN included; results must not change.
        for b, L in enumerate(seq_lens.tolist()):
            inputs[2][b, L:] = float("nan")
            inputs[3][b, L:] = float("nan")
        for num_splits in (1, 3, 64, None):
            o, lse = gva_decode_attention(*inputs, seq_lens, cfg.scale, num_splits)
            name = f"{dtype} splits={num_splits}"
            assert _check(name, o, lse, inputs, seq_lens, cfg.scale, atol), name


def _test_cuda_graph():
    """Capture once, then change lengths and data in place and replay."""
    from gva_reference import GVAConfig

    torch.manual_seed(0)
    dev, dtype, atol = "cuda", torch.bfloat16, 2e-2
    cfg = GVAConfig(d_model=64, num_heads=16, num_groups=4, d_h=128, d_n=128, d_r=16)
    T = 1024  # max cache length for the graph

    for B in (4, 128):
        inputs = _random_inputs(cfg, B, T, dtype, dev)
        seq_lens = torch.ones(B, dtype=torch.int32, device=dev)

        gva_decode_attention(*inputs, seq_lens, cfg.scale)  # warm-up: Triton compiles here
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            o, lse = gva_decode_attention(*inputs, seq_lens, cfg.scale)

        for _ in range(2):
            new_lens = torch.randint(0, T + 1, (B,))
            new_lens[0] = 0  # always one empty request
            seq_lens.copy_(new_lens)
            for t in inputs:
                t.normal_()
            graph.replay()
            name = f"graph B={B} replay"
            assert _check(name, o, lse, inputs, seq_lens, cfg.scale, atol), name


if __name__ == "__main__":
    _test()
    _test_cuda_graph()
    print("ALL PASSED")
