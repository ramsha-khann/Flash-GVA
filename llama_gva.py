"""
A small Llama-style model with GVA attention, generating text with the Flash-GVA kernels.

The weights are random (t), so the generated text would be gibberish. What it shows is the full path a real model takes: embeddings, RMSNorm, RoPE,
SwiGLU, a KV cache, prefill, then one decode step per new token. At the end the same model
is run again with plain PyTorch attention to check that the kernels give the same tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_gva import gva_decode_attention, gva_prefill_attention

DEV, DTYPE = "cuda", torch.bfloat16
VOCAB = 95                 # printable ASCII, one token per character
HIDDEN, FFN, LAYERS = 256, 768, 4
H, G = 8, 2                # 8 query heads, 2 value groups
D_H, D_N, D_R = 64, 32, 32  # value width, key content width, rope width
MAX_LEN = 256              # cache capacity


def encode(text):
    return [ord(c) - 32 for c in text]


def decode(ids):
    return "".join(chr(i + 32) for i in ids)


inv_freq = 1.0 / 10000 ** (torch.arange(0, D_R, 2, device=DEV) / D_R)
angles = torch.outer(torch.arange(MAX_LEN, device=DEV), inv_freq)
COS, SIN = angles.cos(), angles.sin()  # [MAX_LEN, D_R / 2]


def rope(x, pos):
    # x: [B, S, ..., D_R] at positions pos [S]
    shape = (1, len(pos)) + (1,) * (x.dim() - 3) + (D_R // 2,)
    cos, sin = COS[pos].view(shape), SIN[pos].view(shape)
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class GVAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(HIDDEN, H * (D_N + D_R), bias=False)
        self.v = nn.Linear(HIDDEN, G * D_H, bias=False)
        self.k_rope = nn.Linear(HIDDEN, D_R, bias=False)  # one rope key per token, shared by all heads
        self.M = nn.Parameter(torch.randn(H, D_H, D_N) * D_H ** -0.5)  # per-head key map
        self.o = nn.Linear(H * D_H, HIDDEN, bias=False)
        self.scale = (D_N + D_R) ** -0.5

    def forward(self, x, start, cache):
        B, S, _ = x.shape
        pos = torch.arange(start, start + S, device=x.device)
        q_nope, q_rope = self.q(x).view(B, S, H, D_N + D_R).split([D_N, D_R], dim=-1)
        q_rope = rope(q_rope, pos)
        v = self.v(x).view(B, S, G, D_H)
        k_rope = rope(self.k_rope(x), pos)

        if cache is None:
            # Plain PyTorch, keys rebuilt as in the paper: k = [v @ M_h, k_rope]
            v_h = v.repeat_interleave(H // G, dim=2)
            k_nope = torch.einsum("bshd,hdn->bshn", v_h, self.M)
            k = torch.cat([k_nope, k_rope[:, :, None].expand(B, S, H, D_R)], dim=-1)
            q = torch.cat([q_nope, q_rope], dim=-1)
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v_h.transpose(1, 2), is_causal=True, scale=self.scale,
            ).transpose(1, 2)
        else:
            v_cache, k_rope_cache, seqlens = cache
            v_cache[:, start:start + S] = v
            k_rope_cache[:, start:start + S] = k_rope
            q_tilde = torch.einsum("bshn,hdn->bshd", q_nope, self.M)
            if start == 0:
                out = gva_prefill_attention(q_tilde, q_rope, v, k_rope, self.scale)
            else:
                out, _ = gva_decode_attention(q_tilde[:, 0], q_rope[:, 0], v_cache, k_rope_cache, seqlens, self.scale)
                out = out[:, None]

        return self.o(out.reshape(B, S, H * D_H))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = nn.RMSNorm(HIDDEN)
        self.attn = GVAAttention()
        self.mlp_norm = nn.RMSNorm(HIDDEN)
        self.gate = nn.Linear(HIDDEN, FFN, bias=False)
        self.up = nn.Linear(HIDDEN, FFN, bias=False)
        self.down = nn.Linear(FFN, HIDDEN, bias=False)

    def forward(self, x, start, cache):
        x = x + self.attn(self.attn_norm(x), start, cache)
        h = self.mlp_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.RMSNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, tokens, start=0, caches=None):
        # caches=None: plain PyTorch attention over the whole sequence
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            x = block(x, start, None if caches is None else caches[i])
        return self.lm_head(self.norm(x))


@torch.no_grad()
def generate(model, prompt, new_tokens):
    tokens = torch.tensor([encode(prompt)], device=DEV)
    B, S = tokens.shape
    seqlens = torch.full((B,), S, dtype=torch.int32, device=DEV)
    caches = [
        (torch.zeros(B, MAX_LEN, G, D_H, device=DEV, dtype=DTYPE),
         torch.zeros(B, MAX_LEN, D_R, device=DEV, dtype=DTYPE),
         seqlens)
        for _ in model.blocks
    ]

    logits = model(tokens, 0, caches)  # prefill
    tokens = torch.cat([tokens, logits[:, -1:].argmax(-1)], dim=1)
    while tokens.shape[1] < S + new_tokens:
        start = tokens.shape[1] - 1  # position of the newest token
        seqlens.fill_(start + 1)
        logits = model(tokens[:, -1:], start, caches)  # decode
        tokens = torch.cat([tokens, logits[:, -1:].argmax(-1)], dim=1)
    return tokens


if __name__ == "__main__":
    torch.manual_seed(0)
    model = Model().to(DEV, DTYPE).eval()

    prompt = "Grouped value attention"
    tokens = generate(model, prompt, new_tokens=30)
    print("prompt:   ", prompt)
    print("generated:", decode(tokens[0, len(prompt):].tolist()))

    # Same model, plain PyTorch attention, whole sequence at once.
    # Its next-token prediction at every position should match what the kernels generated.
    with torch.no_grad():
        ref_next = model(tokens).argmax(-1)[:, len(prompt) - 1:-1]
    same = (ref_next == tokens[:, len(prompt):]).sum().item()
    print(f"kernels vs plain PyTorch: {same}/{ref_next.shape[1]} tokens match")
    