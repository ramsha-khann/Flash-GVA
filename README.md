# Flash-GVA

Triton prefill and decode kernels for GVA attention, with a plain PyTorch reference to check them against.

- `flash_gva/decode.py` - decode kernel (`gva_decode_attention`)
- `flash_gva/prefill.py` - prefill kernel (`gva_prefill_attention`)
- `flash_gva/reference.py` - PyTorch reference for decode (`gva_decode_attention_ref`)
- `tests/` - compares the kernels with the reference and with PyTorch SDPA

## Setup

Needs Linux and an NVIDIA GPU. Install PyTorch for your CUDA version first, then:

```bash
pip install -e .
```

## Run the tests

```bash
pip install pytest
pytest -v tests/
```

## Use it

```python
from flash_gva import gva_decode_attention, gva_prefill_attention

out = gva_prefill_attention(q_tilde, q_rope, v, k_rope, softmax_scale)                               # prompt
out, lse = gva_decode_attention(q_tilde, q_rope, v_cache, k_rope_cache, cache_seqlens, softmax_scale)  # each new token
```

Shapes: `q_tilde [B, H, d_h]`, `q_rope [B, H, d_r]`, `v_cache [B, T, G, d_h]`, `k_rope_cache [B, T, d_r]`, `cache_seqlens [B]` (int32, on the GPU).
