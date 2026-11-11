"""
Step 3, Part 4 demo: what does the model write before and after a little training?
Run from the mini-llm/ root:  python -m scripts.generate_demo
"""

import time

import numpy as np
import torch

from minillm.model import GPT, GPTConfig
from minillm.tokenizer import Tokenizer

torch.manual_seed(0)
np.random.seed(0)

tok = Tokenizer.load("data/tokenizer.json")
data = np.fromfile("data/fineweb/train_0001.bin", dtype=np.uint16)

config = GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=64)
model = GPT(config)

PROMPT = "The history of the city"


def write(label):
    # Every document in training starts with <|bos|>, so the prompt does too.
    ids = [tok.special["<|bos|>"]] + tok.encode(PROMPT, allow_special=False)
    idx = torch.tensor([ids])
    out = model.generate(idx, max_new_tokens=30, temperature=0.8, top_k=50)
    text = tok.decode(out[0, 1:].tolist())   # [1:] skips <|bos|>
    print(f"\n{label}:\n  {text!r}")


def get_batch(B=16, T=64):
    starts = np.random.randint(0, len(data) - T - 1, B)
    x = np.stack([data[s:s + T] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + T + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


write("untrained (random weights)")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
t0 = time.time()
for step in range(1, 1001):
    x, y = get_batch()
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step in (100, 1000):
        print(f"\n(step {step}, loss {loss.item():.2f}, {time.time() - t0:.0f}s)", end="")
        write(f"after {step} steps")