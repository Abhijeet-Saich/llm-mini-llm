"""
Step 3: The GPT model.
Part 1 of 4: config, embeddings, and the output head.
Part 2 of 4: causal self-attention.

The job of the model: given a sequence of token ids, predict the NEXT token at
every position. Part 1 builds the input and output ends of the model:

    token ids --> [embeddings] --> (Parts 2-3: transformer blocks) --> [output head] --> scores for every token

Without the blocks in the middle, each position only knows its own token, so
this model can only learn "which token usually follows token X" (a "bigram"
model). Attention (Part 2) is what lets positions look at each other.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 16384   # must match the tokenizer
    block_size: int = 1024    # context length: max tokens the model sees at once
    n_layer: int = 12         # number of transformer blocks (Part 3)
    n_head: int = 10          # attention heads per block (Part 2)
    n_embd: int = 640         # size of the vector that represents each token


class CausalSelfAttention(nn.Module):
    """
    Part 2: lets every position look back at earlier positions and pull in
    information from them. "Causal" = a position can never see the future,
    because at that point in the text the future hasn't been written yet.

    For each token we compute three vectors:
      q (query): "what am I looking for?"
      k (key):   "what do I contain?"
      v (value): "what will I hand over if you pick me?"
    Score = how well my query matches your key. High score -> I take more of your value.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must divide evenly into heads"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head   # 640 / 10 = 64

        # One Linear makes q, k and v together (3x wide), then we split it.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # After the heads are glued back together, mix them into one vector.
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.use_flash = True   # fast built-in version; False = the readable version below
        self.att_weights = None  # filled in by the readable version, for inspection

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)            # each (B, T, C)

        # Multi-head: cut each C-sized vector into n_head pieces of head_dim.
        # Each head does its own attention, so different heads can learn to
        # look for different things (the previous word, the subject, a quote mark...).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_flash:
            # Same math as below, done by an optimized PyTorch kernel.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # 1. Score every query against every key: (B, nh, T, T).
            #    Dividing by sqrt(head_dim) keeps scores from growing with vector size.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 2. Causal mask: position i may only look at positions 0..i.
            #    Future positions get -inf, which softmax turns into exactly 0.
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, float("-inf"))
            # 3. Softmax turns each row of scores into weights that sum to 1.
            weights = F.softmax(scores, dim=-1)
            self.att_weights = weights.detach()
            # 4. Each position's output = weighted average of the values.
            y = weights @ v                               # (B, nh, T, hd)

        # Glue the heads back together and mix them.
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.proj(y)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding: a lookup table with one row (a vector of n_embd
        # numbers) per token id. Row 412 is "what the model knows about token 412".
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embedding: one row per position 0..block_size-1, so the
        # model can tell "the cat" from "cat the". Added to the token vector.
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # (Part 3 adds the transformer blocks here)

        # Final normalization, then the output head: turns each n_embd vector
        # back into one score ("logit") per token in the vocabulary.
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input table and the output head share one matrix.
        # Same idea in both directions (token <-> vector), and it saves
        # vocab_size * n_embd parameters (10.5M for our real config).
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Start with small random numbers. Too big and training blows up;
        # all zeros and every unit learns the same thing.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        # idx: token ids, shape (B, T) = (batch size, sequence length)
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence of {T} is longer than block_size"

        pos = torch.arange(T, device=idx.device)       # [0, 1, 2, ..., T-1]
        x = self.wte(idx) + self.wpe(pos)              # (B, T, C) + (T, C) -> (B, T, C)

        # (Part 3: x passes through the transformer blocks here)

        x = self.ln_f(x)
        logits = self.lm_head(x)                       # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy: how surprised the model is by the real next token.
            # Flatten so every position in every sequence is one example.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        # .parameters() lists the shared (tied) matrix only once
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test: python -m minillm.model   (from the mini-llm/ root)
    import numpy as np

    torch.manual_seed(0)
    np.random.seed(0)

    config = GPTConfig(n_embd=128, block_size=64)   # tiny version for a CPU test
    model = GPT(config)
    print(f"parameters: {model.num_params():,}")

    data = np.fromfile("data/fineweb/train_0001.bin", dtype=np.uint16)

    def get_batch(B=16, T=64):
        # Pick B random windows. The target y is x shifted left by one:
        # at every position, the "answer" is the token that comes next.
        starts = np.random.randint(0, len(data) - T - 1, B)
        x = np.stack([data[s:s + T] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1:s + T + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    x, y = get_batch()
    logits, loss = model(x, y)
    print(f"input shape:  {tuple(x.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss at start: {loss.item():.3f}   "
          f"(pure guessing = ln({config.vocab_size}) = {math.log(config.vocab_size):.3f})")

    # A few hundred training steps, just to prove the pieces learn something.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(301):
        x, y = get_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step:3d}: loss {loss.item():.3f}")