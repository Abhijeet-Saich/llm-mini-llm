"""
Checks for Step 3, Part 2 (attention).
Run from the mini-llm/ root:  python -m tests.test_attention
"""

import torch

from minillm.model import CausalSelfAttention, GPTConfig

torch.manual_seed(0)
config = GPTConfig(n_embd=64, n_head=4)
attn = CausalSelfAttention(config)
x = torch.randn(2, 8, 64)   # 2 sequences, 8 positions, 64 numbers per position

# 1. The fast version and the readable version must give the same answer.
attn.use_flash = True
fast = attn(x)
attn.use_flash = False
slow = attn(x)
print(f"1. fast vs readable, biggest difference: {(fast - slow).abs().max().item():.2e}  (should be tiny, ~1e-7)")

# 2. No peeking at the future: change ONLY the last position's input.
#    Outputs at positions 0..6 must not change at all; position 7 should.
x2 = x.clone()
x2[:, -1] = torch.randn(2, 64)
out1, out2 = attn(x), attn(x2)
print(f"2. change at earlier positions: {(out1[:, :-1] - out2[:, :-1]).abs().max().item():.2e}  (should be 0)")
print(f"   change at the last position: {(out1[:, -1] - out2[:, -1]).abs().max().item():.2e}  (should NOT be 0)")

# 3. Look at the attention weights of head 0 on a 5-token sequence.
attn(torch.randn(1, 5, 64))
w = attn.att_weights[0, 0]
print("3. attention weights, head 0 (row = position looking, column = position looked at):")
for i, row in enumerate(w):
    print(f"   pos {i}: " + "  ".join(f"{v:.2f}" for v in row.tolist()) + f"   sum={row.sum().item():.2f}")