"""
Step 4: Feeds batches of tokens to training, straight from the shard files.

Each batch is B random windows of T+1 tokens, picked from anywhere in the
shards. x is the first T tokens of each window, y is the same window shifted
by one (the "next token" answers), exactly like get_batch in model.py's test.
"""

import glob
import os

import numpy as np
import torch


class ShardLoader:
    def __init__(self, data_dir, split, batch_size, block_size, seed=0):
        self.files = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
        assert self.files, f"no {split}_*.bin files found in {data_dir}"

        # memmap = "memory map": the file stays on disk and numpy reads only the
        # bytes we actually touch. 2 billion tokens (4 GB) never has to fit in RAM.
        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]

        # Bigger shards should be picked more often, so every token has the
        # same chance of being chosen no matter which file it's in.
        sizes = np.array([len(s) for s in self.shards], dtype=np.float64)
        self.shard_probs = sizes / sizes.sum()
        self.num_tokens = int(sizes.sum())

        self.B, self.T = batch_size, block_size
        self.seed = seed
        self.reset()

    def reset(self, seed=None):
        """Restart the random number generator. Calling this before every
        evaluation means we measure val loss on the SAME batches each time,
        so the numbers are comparable from one evaluation to the next."""
        self.rng = np.random.default_rng(self.seed if seed is None else seed)

    def next_batch(self, device="cpu"):
        B, T = self.B, self.T
        buf = np.empty((B, T + 1), dtype=np.int64)
        shard_ids = self.rng.choice(len(self.shards), size=B, p=self.shard_probs)
        for row, si in enumerate(shard_ids):
            shard = self.shards[si]
            start = self.rng.integers(0, len(shard) - T - 1)
            buf[row] = shard[start:start + T + 1]

        batch = torch.from_numpy(buf)
        if device.startswith("cuda"):
            # "Pinned" memory can be copied to the GPU in the background
            # while the GPU is still busy with the previous step.
            batch = batch.pin_memory().to(device, non_blocking=True)

        x = batch[:, :-1].contiguous()
        y = batch[:, 1:].contiguous()
        return x, y