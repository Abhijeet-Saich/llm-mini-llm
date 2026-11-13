"""
Step 6, Part 2: chat tuning (SFT = supervised fine-tuning).

Starts from a pretrained checkpoint and trains on real conversations from the
smol-smoltalk dataset, learning ONLY from the assistant's tokens (the mask
from chat_format.py). Afterwards the model answers instead of continuing text.

On Modal (Step 6, Part 3), e.g. for the rehearsal model:
    ... --script scripts.sft --args "--init /vol/checkpoints/rehearsal/ckpt.pt --out-dir /vol/checkpoints/rehearsal-sft"
"""

import argparse
import contextlib
import os
import random
import time
from dataclasses import asdict
from multiprocessing import Pool

import torch

from minillm.chat_format import render_conversation, render_for_reply
from minillm.model import GPT, GPTConfig
from minillm.tokenizer import Tokenizer

# Fixed questions, asked during training so you can watch the model learn to answer.
TEST_PROMPTS = [
    "What is the capital of France?",
    "Give me three tips for staying focused while studying.",
    "Explain what a black hole is in simple words.",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="pretrained checkpoint to start from")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out-dir", default="checkpoints/sft")
    p.add_argument("--max-examples", type=int, default=0, help="0 = use every conversation")
    p.add_argument("--val-examples", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="0 = one full pass per epoch")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)      # 64 conversations per step
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------- data
def load_conversations(max_examples):
    """smol-smoltalk: ~460k conversations written for training small models."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    convos = []
    for row in ds:
        convos.append(row["messages"])
        if max_examples and len(convos) >= max_examples:
            break
    return convos


_tok, _max_len = None, None


def _init_worker(tokenizer_path, max_len):
    global _tok, _max_len
    _tok, _max_len = Tokenizer.load(tokenizer_path), max_len


def _render(messages):
    try:
        return render_conversation(_tok, messages, max_len=_max_len)
    except ValueError:      # a role we don't support: skip this conversation
        return None


def make_batch(examples, device):
    """Pad a list of (ids, mask) to the longest one in the batch.
    x = tokens 0..n-1, y = tokens 1..n (the next-token answers), exactly as in
    pretraining, except y is -100 wherever the mask says "don't learn this".
    F.cross_entropy skips every -100 (its default ignore_index), so those
    positions add nothing to the loss. Padding goes at the END of each row,
    and causal attention never looks forward, so real tokens never see it."""
    T = max(len(ids) for ids, _ in examples) - 1
    x = torch.zeros((len(examples), T), dtype=torch.long)
    y = torch.full((len(examples), T), -100, dtype=torch.long)
    for row, (ids, mask) in enumerate(examples):
        n = len(ids) - 1
        x[row, :n] = torch.tensor(ids[:-1])
        target = torch.tensor(ids[1:])
        target[~torch.tensor(mask[1:], dtype=torch.bool)] = -100
        y[row, :n] = target
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device == "cuda" else contextlib.nullcontext())

    # 1. The pretrained model.
    ckpt = torch.load(args.init, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    del ckpt
    tok = Tokenizer.load(args.tokenizer)
    stop = tok.special["<|assistant_end|>"]
    print(f"loaded {args.init}: {model.num_params():,} parameters, context {config.block_size}")

    # 2. Conversations -> (ids, mask), tokenized in parallel.
    t0 = time.time()
    convos = load_conversations(args.max_examples)
    max_len = config.block_size + 1          # +1 because x and y are shifted by one
    with Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer, max_len)) as pool:
        rendered = pool.map(_render, convos, chunksize=256)
    # Keep only conversations with something to learn after cutting to max_len.
    examples = [r for r in rendered if r is not None and sum(r[1][1:]) > 0]
    random.shuffle(examples)
    val, train = examples[:args.val_examples], examples[args.val_examples:]
    n_tokens = sum(len(ids) for ids, _ in train)
    n_learn = sum(sum(m) for _, m in train)
    print(f"{len(convos):,} conversations -> {len(train):,} train, {len(val):,} val "
          f"({time.time() - t0:.0f}s)")
    print(f"train tokens: {n_tokens / 1e6:.1f}M, of which learned from: {n_learn / 1e6:.1f}M "
          f"({100 * n_learn / n_tokens:.0f}%)")

    per_step = args.batch_size * args.grad_accum
    steps_per_epoch = len(train) // per_step
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    print(f"{per_step} conversations per step, {total_steps:,} steps")

    # 3. Optimizer: a fresh AdamW, gentler than pretraining (we're adjusting, not starting over).
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.0, fused=(device == "cuda"))

    def get_lr(step):
        # Short warmup, then a straight line down to zero.
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        return args.lr * max(0.0, 1 - (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))

    @torch.no_grad()
    def evaluate():
        model.eval()
        total, n = 0.0, 0
        for i in range(0, len(val), args.batch_size):
            x, y = make_batch(val[i:i + args.batch_size], device)
            with autocast:
                _, loss = model(x, y)
            total += loss.item()
            n += 1
        model.train()
        return total / max(1, n)

    def chat_samples():
        model.eval()
        for prompt in TEST_PROMPTS:
            ids = render_for_reply(tok, [{"role": "user", "content": prompt}])
            idx = torch.tensor([ids], device=device)
            with autocast:
                out = model.generate(idx, max_new_tokens=120, temperature=0.7, top_k=50,
                                     stop_token=stop)
            reply = [i for i in out[0, len(ids):].tolist() if i != stop]
            print(f"  Q: {prompt}\n  A: {tok.decode(reply)!r}")
        model.train()

    def save(step):
        ckpt = {"model": model.state_dict(), "config": asdict(config),
                "step": step, "args": vars(args)}
        path = os.path.join(args.out_dir, "ckpt.pt")
        torch.save(ckpt, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"  saved checkpoint at step {step}")

    print(f"\nbefore chat tuning: val loss {evaluate():.3f}")
    chat_samples()

    # 4. The loop: same shape as pretraining, but batches are whole conversations.
    model.train()
    order, cursor = [], 0
    for step in range(total_steps):
        t0 = time.time()
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            if cursor + args.batch_size > len(order):        # start a new pass
                order, cursor = random.sample(range(len(train)), len(train)), 0
            batch = [train[i] for i in order[cursor:cursor + args.batch_size]]
            cursor += args.batch_size
            x, y = make_batch(batch, device)
            with autocast:
                _, loss = model(x, y)
            loss = loss / args.grad_accum
            loss_sum += loss.item()
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        dt = time.time() - t0

        done = step + 1
        last = done == total_steps
        if done % args.log_every == 0 or last or step == 0:
            print(f"step {done:5d}/{total_steps} | loss {loss_sum:.3f} | lr {lr:.2e} | "
                  f"grad norm {grad_norm:.2f} | {dt * 1000:.0f} ms", flush=True)
        if done % args.eval_every == 0 or last:
            print(f"  val loss {evaluate():.3f}", flush=True)
        if done % args.sample_every == 0 or last:
            chat_samples()
        if done % args.save_every == 0 or last:
            save(done)


if __name__ == "__main__":
    main()