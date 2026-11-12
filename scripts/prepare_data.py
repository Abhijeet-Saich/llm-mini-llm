"""
Step 2b: Tokenize FineWeb-Edu and save it as binary "shard" files.

Run from the mini-llm/ root folder (small test run):
    python -m scripts.prepare_data --max-tokens 20000000

Output, in data/fineweb/:
    val_0000.bin     first shard: validation data (never trained on)
    train_0001.bin   training shards...
    train_0002.bin

Each file is just token ids, one after another, as 2-byte unsigned ints
(uint16, which holds 0-65535, enough for our 16,384 vocab). Every document
starts with <|bos|>, so the model can tell where one document ends and the
next begins. There is no other structure: training will read random windows
straight out of these files.
"""

import argparse
import os
import time
from itertools import islice
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from minillm.fineweb import iter_documents
from minillm.tokenizer import Tokenizer
import sys 

# Each worker process loads its own copy of the tokenizer once, here.
tokenizer = None


def init_worker(tokenizer_path):
    global tokenizer
    tokenizer = Tokenizer.load(tokenizer_path)


def tokenize_doc(text):
    ids = [tokenizer.special["<|bos|>"]]
    # allow_special=False: web text containing "<|bos|>" is treated as plain text
    ids.extend(tokenizer.encode(text, allow_special=False))
    return np.array(ids, dtype=np.uint16)


def batches(iterable, size):
    """Group an iterator into lists of `size` items. Feeding the pool in batches
    keeps memory flat (pool.imap would read the whole dataset into memory)."""
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--out-dir", default="data/fineweb")
    parser.add_argument("--max-tokens", type=int, default=20_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--shard-tokens", type=int, default=50_000_000)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vocab_size = Tokenizer.load(args.tokenizer).vocab_size
    assert vocab_size <= 2**16, "vocab too big for uint16"

    shard_index = 0
    capacity = args.val_tokens                      # shard 0 (val) is smaller
    buffer = np.empty(args.shard_tokens, dtype=np.uint16)
    filled = 0
    total = 0

    def write_shard():
        name = "val" if shard_index == 0 else "train"
        path = os.path.join(args.out_dir, f"{name}_{shard_index:04d}.bin")
        buffer[:filled].tofile(path)
        tqdm.write(f"wrote {path} ({filled:,} tokens)")

    t0 = time.time()
    bar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True)
    with Pool(args.workers, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        for batch in batches(iter_documents(), 256):
            for ids in pool.map(tokenize_doc, batch, chunksize=8):
                # Copy this document's tokens into the buffer. A document may be
                # split across two shards; that's fine, training reads a stream.
                pos = 0
                while pos < len(ids):
                    take = min(capacity - filled, len(ids) - pos)
                    buffer[filled:filled + take] = ids[pos:pos + take]
                    filled += take
                    pos += take
                    if filled == capacity:
                        write_shard()
                        shard_index += 1
                        capacity = args.shard_tokens
                        filled = 0
                total += len(ids)
                bar.update(len(ids))
            if total >= args.max_tokens:
                break
    bar.close()

    if filled > 0:
        write_shard()
    minutes = (time.time() - t0) / 60
    print(f"done: {total:,} tokens in {minutes:.1f} min "
          f"({total / (time.time() - t0):,.0f} tokens/sec)")


if __name__ == "__main__":
    main()
    # The streaming download leaves background threads running. All our files
    # are already written, so exit right away instead of letting Python's
    # normal shutdown trip over those threads.
    sys.stdout.flush()
    os._exit(0)