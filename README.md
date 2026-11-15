# Flash-GVA

Triton decode kernel for GVA attention, with a plain PyTorch reference to check it against.

- `gva_decode_triton.py` - the Triton kernel (`gva_decode_attention`)
- `gva_reference.py` - PyTorch reference (`gva_decode_attention_ref`)
- `tests/test_gva_decode.py` - compares the kernel with the reference

## Setup

Needs an NVIDIA GPU with CUDA 12.8 drivers.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt --no-build-isolation
```

## Run the tests

```bash
pytest -v tests/
```

## Use it

```python
from gva_decode_triton import gva_decode_attention

out, lse = gva_decode_attention(q_tilde, q_rope, v_cache, k_rope_cache, cache_seqlens, softmax_scale)
```

Shapes: `q_tilde [B, H, d_h]`, `q_rope [B, H, d_r]`, `v_cache [B, T, G, d_h]`, `k_rope_cache [B, T, d_r]`, `cache_seqlens [B]` (int32, on the GPU).
