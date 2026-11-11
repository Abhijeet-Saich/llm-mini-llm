"""
Step 4: The pretraining loop.

Local test on your Mac (tiny model, CPU, a few minutes):
    python -m scripts.pretrain --preset tiny

Continue a run that stopped (crash, Ctrl+C, Modal timeout):
    python -m scripts.pretrain --preset tiny --resume

The real run on Modal uses --preset full (Step 5).
Any setting can be overridden, e.g.  --max-steps 1000 --lr 1e-3
"""

import argparse
import contextlib
import math
import os
import time
from dataclasses import asdict

import torch

from minillm.dataloader import ShardLoader
from minillm.model import GPT, GPTConfig
from minillm.tokenizer import Tokenizer

# Two sets of defaults. "tiny" runs on a laptop CPU; "full" is the real model.
PRESETS = {
    "tiny": dict(
        n_layer=2, n_head=4, n_embd=128, block_size=64,
        batch_size=16, grad_accum=1,
        max_steps=500, warmup_steps=50, lr=3e-3,
        eval_every=100, eval_batches=20, sample_every=250, save_every=250, log_every=25,
    ),
    "full": dict(
        n_layer=12, n_head=10, n_embd=640, block_size=1024,
        batch_size=32, grad_accum=16,          # 32 x 1024 x 16 = 524,288 tokens per step
        max_steps=4000, warmup_steps=200, lr=6e-4,   # we'll finalize these in Step 5
        eval_every=250, eval_batches=50, sample_every=500, save_every=250, log_every=10,
    ),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default="tiny")
    p.add_argument("--data-dir", default="data/fineweb")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out-dir", default="checkpoints/pretrain")
    p.add_argument("--resume", action="store_true", help="continue from out-dir/ckpt.pt")
    p.add_argument("--compile", action="store_true", help="torch.compile (GPU only, faster)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    # Everything below defaults to the preset's value unless you pass it.
    for name in PRESETS["tiny"]:
        kind = float if name == "lr" else int
        p.add_argument("--" + name.replace("_", "-"), type=kind, default=None)
    args = p.parse_args()
    for name, value in PRESETS[args.preset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def get_lr(step, args):
    """Learning-rate schedule: warm up, then cosine decay.
    - Warmup: start tiny and ramp up linearly. At step 0 the weights are random
      and the gradients are wild; a big learning rate right away can wreck the model.
    - Cosine decay: then glide smoothly down to 10% of the peak. Big steps early
      to learn fast, small steps late to settle into a good spot."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    progress = min(progress, 1.0)
    min_lr = args.lr * args.min_lr_frac
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - min_lr)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    log_path = os.path.join(args.out_dir, "log.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True   # faster matmuls on modern GPUs
    # bfloat16 on GPU: half the memory and much faster, with nearly the same accuracy.
    # On CPU we stay in normal float32.
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device == "cuda" else contextlib.nullcontext())

    tok = Tokenizer.load(args.tokenizer)

    # ---- model (fresh, or restored from a checkpoint) ----
    ckpt = None
    if args.resume:
        assert os.path.exists(ckpt_path), f"--resume given but {ckpt_path} doesn't exist"
        ckpt = torch.load(ckpt_path, map_location=device)
        config = GPTConfig(**ckpt["config"])
    else:
        config = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                           n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(config).to(device)
    if ckpt:
        model.load_state_dict(ckpt["model"])
    raw_model = model          # keep the plain model for saving and sampling
    if args.compile:
        model = torch.compile(model)

    # ---- optimizer ----
    # Weight decay gently pulls weights toward zero, which helps prevent
    # overfitting. Only apply it to the big matrices (2D), not to the small
    # LayerNorm vectors (1D), which need to stay free to scale things.
    params = [p for p in raw_model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": args.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95),
                                  fused=(device == "cuda"))
    start_step = 0
    if ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from {ckpt_path} at step {start_step}")
        del ckpt

    # ---- data ----
    # Seed the training loader with the step we start at, so a resumed run
    # doesn't replay the exact same batches it already trained on.
    train_loader = ShardLoader(args.data_dir, "train", args.batch_size, args.block_size,
                               seed=args.seed + start_step)
    val_loader = ShardLoader(args.data_dir, "val", args.batch_size, args.block_size, seed=1234)

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    total_tokens = tokens_per_step * args.max_steps
    print(f"device: {device} | model: {raw_model.num_params():,} parameters")
    print(f"tokens per step: {tokens_per_step:,} | steps: {args.max_steps:,} | "
          f"total: {total_tokens / 1e6:,.1f}M tokens")
    print(f"training data: {train_loader.num_tokens / 1e6:,.1f}M tokens "
          f"(this run sees each token about {total_tokens / train_loader.num_tokens:.2f} times)")

    def log(step, kind, value):
        with open(log_path, "a") as f:
            f.write(f"{step},{kind},{value:.4f}\n")

    @torch.no_grad()
    def evaluate():
        model.eval()
        val_loader.reset()        # same val batches every time
        total = 0.0
        for _ in range(args.eval_batches):
            x, y = val_loader.next_batch(device)
            with autocast:
                _, loss = model(x, y)
            total += loss.item()
        model.train()
        return total / args.eval_batches

    def sample(prompt="The history of the city"):
        ids = [tok.special["<|bos|>"]] + tok.encode(prompt, allow_special=False)
        idx = torch.tensor([ids], device=device)
        with autocast:
            out = raw_model.generate(idx, max_new_tokens=40, temperature=0.8, top_k=50)
        return tok.decode(out[0, 1:].tolist())

    def save(step):
        ckpt = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "config": asdict(config), "args": vars(args)}
        # Write to a temporary file, then rename. If we crash mid-save,
        # the previous good checkpoint is still intact.
        torch.save(ckpt, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        print(f"  saved checkpoint at step {step}")

    # ---- the loop ----
    model.train()
    for step in range(start_step, args.max_steps):
        t0 = time.time()
        lr = get_lr(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Gradient accumulation: run several small batches and add up their
        # gradients before taking one optimizer step. Same result as one big
        # batch, without needing the memory for it.
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for _ in range(args.grad_accum):
            x, y = train_loader.next_batch(device)
            with autocast:
                _, loss = model(x, y)
            loss = loss / args.grad_accum     # average, not sum, across micro-batches
            loss_sum += loss.detach()
            loss.backward()

        # Gradient clipping: if one unlucky batch produces a huge gradient,
        # shrink it so a single step can't throw the model far off course.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()          # wait for the GPU so timing is honest
        dt = time.time() - t0

        done = step + 1
        last = done == args.max_steps
        if done % args.log_every == 0 or last or step == start_step:
            train_loss = loss_sum.item()
            print(f"step {done:5d}/{args.max_steps} | loss {train_loss:.3f} | lr {lr:.2e} | "
                  f"grad norm {grad_norm:.2f} | {dt * 1000:.0f} ms | {tokens_per_step / dt:,.0f} tok/s")
            log(done, "train", train_loss)

        if done % args.eval_every == 0 or last:
            val_loss = evaluate()
            print(f"  val loss {val_loss:.3f}")
            log(done, "val", val_loss)

        if done % args.sample_every == 0 or last:
            print(f"  sample: {sample()!r}")

        if done % args.save_every == 0 or last:
            save(done)


if __name__ == "__main__":
    main()