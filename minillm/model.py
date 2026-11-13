"""
Step 3: The GPT model.
Part 1 of 4: config, embeddings, and the output head.
Part 2 of 4: causal self-attention.
Part 3 of 4: the MLP, the transformer block, and the full model.
Part 4 of 4: generating text.

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


class MLP(nn.Module):
    """
    Part 3: after attention GATHERS information from other positions, the MLP
    PROCESSES it, separately at each position. Widen to 4x, apply a non-linear
    function (GELU), shrink back. Most of the model's "knowledge" (facts,
    word patterns) is thought to be stored in these weights.
    """

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """
    One transformer block = attention + MLP, each wrapped the same way:
        x = x + layer(norm(x))
    The "x +" is the residual connection: each layer ADDS a small change to x
    instead of replacing it. Information (and gradients during training) can
    flow straight through all 12 blocks, which is what makes deep models trainable.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # positions talk to each other
        x = x + self.mlp(self.ln_2(x))    # each position thinks on its own
        return x


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

        # The transformer blocks, stacked n_layer deep.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final normalization, then the output head: turns each n_embd vector
        # back into one score ("logit") per token in the vocabulary.
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input table and the output head share one matrix.
        # Same idea in both directions (token <-> vector), and it saves
        # vocab_size * n_embd parameters (10.5M for our real config).
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

        # Each block ADDS its output to x, so with 12 blocks x would grow with
        # depth. Starting the two "proj" layers (the last layer before each
        # addition) extra small keeps x's size steady at the start of training.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

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

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                       # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy: how surprised the model is by the real next token.
            # Flatten so every position in every sequence is one example.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()   # no training here, so skip gradient bookkeeping
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        """
        Part 4: write text, one token at a time.
        idx: (B, T) token ids to start from (the "prompt").
        Each loop: run the model, look ONLY at the scores for the last position
        (that's the prediction for the next token), pick a token, append it,
        and feed the longer sequence back in.
        """
        for _ in range(max_new_tokens):
            # The model can't see more than block_size tokens, so keep the most recent.
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                        # (B, vocab_size)

            if temperature == 0:
                # Greedy: always take the single most likely token.
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                # Temperature: < 1 makes likely tokens even more likely (safer,
                # more repetitive); > 1 flattens the odds (more random).
                logits = logits / temperature
                if top_k is not None:
                    # Only allow the top_k most likely tokens; ban the long tail of junk.
                    kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits[logits < kth_best] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # roll a weighted die

            idx = torch.cat([idx, next_id], dim=1)
            # Step 6: stop early once the model writes the stop token (e.g. <|assistant_end|>).
            if stop_token is not None and (next_id == stop_token).all():
                break
        return idx

    def num_params(self):
        # .parameters() lists the shared (tied) matrix only once
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test: python -m minillm.model   (from the mini-llm/ root)
    import time

    import numpy as np

    real = GPT(GPTConfig())
    print(f"real model (12 layers, 640 wide): {real.num_params():,} parameters")
    del real

    data = np.fromfile("data/fineweb/train_0001.bin", dtype=np.uint16)

    def get_batch(B=16, T=64):
        # Pick B random windows. The target y is x shifted left by one:
        # at every position, the "answer" is the token that comes next.
        starts = np.random.randint(0, len(data) - T - 1, B)
        x = np.stack([data[s:s + T] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1:s + T + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def train_tiny(n_layer, steps=300):
        # Same seed and same batches for both runs, so it's a fair comparison.
        torch.manual_seed(0)
        np.random.seed(0)
        config = GPTConfig(n_layer=n_layer, n_head=4, n_embd=128, block_size=64)
        model = GPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        print(f"\n--- {n_layer} blocks, {model.num_params():,} parameters ---")
        recent = []
        t0 = time.time()
        for step in range(steps + 1):
            x, y = get_batch()
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            recent = (recent + [loss.item()])[-50:]
            if step % 100 == 0:
                print(f"step {step:3d}: loss {loss.item():.3f}")
        avg = sum(recent) / len(recent)
        print(f"average loss over last 50 steps: {avg:.3f}  ({time.time() - t0:.0f}s)")
        return avg

    # 0 blocks = Part 1's bigram model. 2 blocks = a real (tiny) transformer.
    bigram = train_tiny(n_layer=0)
    transformer = train_tiny(n_layer=2)
    print(f"\ntransformer beats bigram by {bigram - transformer:.3f}")