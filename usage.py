import torch
from flash_gva import gva_prefill_attention, gva_decode_attention

B, H, G = 2, 16, 4            # 16 query heads share 4 value groups (head h uses group h // 4)
d_h, d_n, d_r = 128, 128, 64  # value width, key content width, rope width
hidden = 2048                 # model width
T = 4096                      # cache capacity per request
dev, dt = "cuda", torch.bfloat16
scale = (d_n + d_r) ** -0.5

# Model weights. Random here, from the checkpoint in a real model.
M = torch.randn(H, d_h, d_n, device=dev, dtype=dt)            # per-head key map
W_o = torch.randn(H * d_h, hidden, device=dev, dtype=dt) * 0.02  # output projection

# Allocated once, before the first request.
v_cache = torch.zeros(B, T, G, d_h, device=dev, dtype=dt)
k_rope_cache = torch.zeros(B, T, d_r, device=dev, dtype=dt)
cache_seqlens = torch.zeros(B, dtype=torch.int32, device=dev)

# Prefill: the whole prompt at once (every request has the same length S).
# In a model these come from the layer's projections, with RoPE already applied.
S = 100
q_nope = torch.randn(B, S, H, d_n, device=dev, dtype=dt)
q_rope = torch.randn(B, S, H, d_r, device=dev, dtype=dt)
v = torch.randn(B, S, G, d_h, device=dev, dtype=dt)
k_rope = torch.randn(B, S, d_r, device=dev, dtype=dt)

q_tilde = torch.einsum("bshn,hdn->bshd", q_nope, M)  # fold M into the query
out = gva_prefill_attention(q_tilde, q_rope, v, k_rope, scale)  # [B, S, H, d_h]
layer_out = out.reshape(B, S, H * d_h) @ W_o                    # [B, S, hidden], goes to the next layer

v_cache[:, :S] = v  # keep the prompt for decoding
k_rope_cache[:, :S] = k_rope
cache_seqlens.fill_(S)

# Decode: one new token per request, repeated until generation stops.
for step in range(3):
    q_nope_t = torch.randn(B, H, d_n, device=dev, dtype=dt)
    q_rope_t = torch.randn(B, H, d_r, device=dev, dtype=dt)
    v_t = torch.randn(B, G, d_h, device=dev, dtype=dt)
    k_rope_t = torch.randn(B, d_r, device=dev, dtype=dt)

    rows = torch.arange(B, device=dev)
    v_cache[rows, cache_seqlens.long()] = v_t  # request b writes at row cache_seqlens[b]
    k_rope_cache[rows, cache_seqlens.long()] = k_rope_t
    cache_seqlens += 1

    q_tilde_t = torch.einsum("bhn,hdn->bhd", q_nope_t, M)
    # lse is only needed to combine partial results (e.g. a cache split across GPUs).
    out_t, _ = gva_decode_attention(q_tilde_t, q_rope_t, v_cache, k_rope_cache, cache_seqlens, scale)  # [B, H, d_h]
    layer_out = out_t.reshape(B, H * d_h) @ W_o                                                      # [B, hidden]

print("prefill out", tuple(out.shape), "decode out", tuple(out_t.shape), "layer out", tuple(layer_out.shape))
