# mini-llm

A small language model built from scratch, one step at a time.

This README covers the steps finished so far: **building a tokenizer** and **turning a large text dataset into token files** that a model can train on.

| Step | What it does | Code |
|------|--------------|------|
| 1  | Write a byte-level BPE tokenizer from scratch | [minillm/tokenizer.py](minillm/tokenizer.py) |
| 2a | Train that tokenizer on real web text (FineWeb-Edu) | [scripts/train_tokenizer.py](scripts/train_tokenizer.py) |
| 2b | Tokenize the dataset and save it as binary shard files | [scripts/prepare_data.py](scripts/prepare_data.py) |

---

## Quick start

```bash
# 1. Create and activate a virtual environment (not committed to git)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the tokenizer       -> data/tokenizer.json
python -m scripts.train_tokenizer

# 4. Tokenize the dataset      -> data/fineweb/*.bin
python -m scripts.prepare_data --max-tokens 20000000
```

Run every command from the project root (`mini-llm/`). The `-m` flag runs the scripts as modules, which lets them import from `minillm/`.

---

## Why we need a tokenizer

A neural network can't read text, only numbers. A **tokenizer** converts text into a list of integers, called **token ids**, and can turn those ids back into the original text:

```
"Hello there!"  --encode-->  [72, 12114, 631, 33]  --decode-->  "Hello there!"
                              'H'  'ello' ' there' '!'
```
(These are the actual ids from our trained tokenizer. `"Hello"` is split as `H` + `ello`, probably because a capitalized "Hello" is rare in educational web text.)

The simplest approach uses one number per character or byte. That works, but it produces very long sequences, so the model sees less text per training step. **BPE (Byte Pair Encoding)** fixes this by learning to merge common pieces of text, like `"th"`, `"ing"` or `" the"`, into single tokens.

---

## Step 1: The BPE tokenizer ([minillm/tokenizer.py](minillm/tokenizer.py))

### 1.1 Start from bytes
Text is stored as UTF-8 bytes, and each byte is a number from 0 to 255. So the tokenizer starts with **256 base tokens**, one per byte. This means it can encode *any* text, including emoji and other languages, without ever hitting an "unknown" token.

### 1.2 Split text into chunks first
Before merging, the text is split into rough word-like chunks using the same regex pattern as GPT-4 (`SPLIT_PATTERN`):

```
"Hello there, it's 2024!"  ->  ["Hello", " there", ",", " it", "'s", " ", "202", "4", "!"]
```

Merges never cross a chunk border. That stops the tokenizer from learning odd tokens like `"dog."` that mix words and punctuation. Numbers are split into groups of at most 3 digits.

### 1.3 Training: repeatedly merge the most common pair
1. Count every unique chunk. (`"the"` might appear 50,000 times, but it only needs to be processed once.)
2. Count every pair of adjacent tokens across all chunks, weighted by how often each chunk appears.
3. Take the **most frequent pair**, e.g. `(t, h)`, give it a new id (256, 257, ...), and replace it everywhere.
4. Repeat until the vocabulary reaches the target size.

**Speed trick:** a `where` index records which chunks contain each pair. After a merge, only those chunks are updated, so nothing gets rescanned from scratch. That's what makes training on 20M characters practical in pure Python.

### 1.4 Special tokens
Five extra tokens get the final ids. They never come from normal text and are used to mark structure:

| Token | Purpose |
|-------|---------|
| `<\|bos\|>` | Beginning of a document |
| `<\|user_start\|>` / `<\|user_end\|>` | Wrap a user's message (later, for chat) |
| `<\|assistant_start\|>` / `<\|assistant_end\|>` | Wrap the model's reply (later, for chat) |

### 1.5 Encoding and decoding
- **`encode(text)`**: splits text into chunks, converts each chunk to bytes, then applies the learned merges **in the order they were learned**. A cache of up to 500,000 chunks speeds up common words.
- **`decode(ids)`**: looks up the bytes for each id, joins them, and converts back to a string. `errors="replace"` stops a half-finished multi-byte character from crashing it.
- **`save(path)` / `Tokenizer.load(path)`**: store the merges and special tokens as JSON.

### Try it on any text file
```bash
python minillm/tokenizer.py some_text.txt 1024
```
This trains a small 1,024-token vocabulary, saves `tokenizer.json`, and checks that encode → decode gives back the exact original text.

---

## Step 2a: Train the tokenizer on real data ([scripts/train_tokenizer.py](scripts/train_tokenizer.py))

```bash
python -m scripts.train_tokenizer
```

**Data source:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`), high-quality educational web pages filtered by Hugging Face. [minillm/fineweb.py](minillm/fineweb.py) **streams** documents one at a time, so the full 10-billion-token dataset is never downloaded.

What the script does:
1. Streams documents until it has **20M characters for training** plus **1M held-out characters for testing**.
2. Trains the tokenizer to a **vocabulary of 16,384** (256 bytes + 16,123 merges + 5 special tokens).
3. Saves it to `data/tokenizer.json`.
4. Reports **compression** (bytes per token) on the held-out text. Higher is better, because the same text becomes fewer tokens. It also checks that decoding gives back the exact original text.
5. Prints the last tokens it learned and shows an example sentence split into tokens.

| Option | Default | Meaning |
|--------|---------|---------|
| `--vocab-size` | `16384` | Total number of tokens |
| `--train-chars` | `20000000` | Characters of text to learn merges from |
| `--test-chars` | `1000000` | Held-out characters for measuring compression |
| `--out` | `data/tokenizer.json` | Where to save the tokenizer |

Our trained tokenizer: `<|bos|>` = 16379 and `<|assistant_end|>` = 16383, the last id.

---

## Step 2b: Tokenize the dataset into shards ([scripts/prepare_data.py](scripts/prepare_data.py))

```bash
python -m scripts.prepare_data --max-tokens 20000000
```

This turns raw text into the format the model will train on:

1. **Each document starts with `<|bos|>`**, so the model can tell where one document ends and the next begins.
2. **Documents are tokenized in parallel** with a pool of worker processes, one per CPU core by default. Each worker loads its own copy of the tokenizer. Documents are fed in batches of 256 to keep memory use flat.
3. **Tokens are stored as `uint16`** (2 bytes each, range 0–65,535), which is enough for our 16,384-token vocabulary.
4. **Tokens are written into shard files.** The first shard is a smaller **validation** set that is never trained on. The rest are **training** shards. A document may be split across two shards, which is fine because training reads one continuous stream of tokens.

Output in `data/fineweb/`:

| File | Tokens | Size |
|------|--------|------|
| `val_0000.bin` | 2,000,000 | 4.0 MB |
| `train_0001.bin` | 18,232,204 | 36.5 MB |

Each file is just token ids laid end to end, with no header. Training will read random windows directly from these files.

| Option | Default | Meaning |
|--------|---------|---------|
| `--tokenizer` | `data/tokenizer.json` | Tokenizer to use |
| `--out-dir` | `data/fineweb` | Where to write shards |
| `--max-tokens` | `20000000` | Stop after roughly this many tokens |
| `--val-tokens` | `2000000` | Size of the validation shard |
| `--shard-tokens` | `50000000` | Size of each training shard |
| `--workers` | all CPU cores | Number of parallel processes |

### Reading a shard back
```python
import numpy as np
from minillm.tokenizer import Tokenizer

tok = Tokenizer.load("data/tokenizer.json")
ids = np.fromfile("data/fineweb/val_0000.bin", dtype=np.uint16)
print(len(ids), "tokens")
print(tok.decode(ids[:100].tolist()))
```

---

## Project layout

```
mini-llm/
├── minillm/
│   ├── tokenizer.py        # Step 1: BPE tokenizer
│   └── fineweb.py          # streams FineWeb-Edu documents
├── scripts/
│   ├── train_tokenizer.py  # Step 2a
│   └── prepare_data.py     # Step 2b
├── data/
│   ├── tokenizer.json      # trained tokenizer
│   └── fineweb/            # tokenized shards (.bin)
├── checkpoints/            # (upcoming) saved model weights
├── modal_app/              # (upcoming) cloud training on Modal
├── tests/
├── web/
└── requirements.txt
```

## Next steps
- Build a data loader that samples random windows from the `.bin` shards
- Write the transformer model
- Train it and save checkpoints
