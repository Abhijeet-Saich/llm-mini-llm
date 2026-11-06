"""
Step 1: A byte-level BPE (Byte Pair Encoding) tokenizer, written from scratch.

Big idea:
  - Computers store text as bytes (numbers 0-255). So we start with 256 "tokens".
  - We look for the pair of tokens that appears most often next to each other
    (e.g. "t" + "h") and give that pair a brand-new token id (256).
  - Repeat thousands of times. Common pieces of words become single tokens,
    so text turns into far fewer numbers -> the model sees more text per step.
"""

import json
from collections import Counter, defaultdict

import regex as re  # pip install regex  (supports \p{L} = "any letter")

# Before BPE, we split text into chunks (roughly words, numbers, punctuation).
# Merges never cross chunk borders, so "dog." never becomes one weird token.
# This is the same splitting pattern GPT-4 uses.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Special tokens are never produced from normal text. We use them later to mark
# the start of a document and who is speaking in a chat.
SPECIAL_TOKENS = [
    "<|bos|>",              # beginning of a document
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
]


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in the list `ids` with `new_id`.
    Example: merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]"""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self):
        self.merges = {}                                   # (id_a, id_b) -> new_id
        self.vocab = {i: bytes([i]) for i in range(256)}   # id -> the bytes it stands for
        self.special = {}                                  # "<|bos|>" -> id
        self.pattern = re.compile(SPLIT_PATTERN)
        self.cache = {}                                    # chunk text -> ids (speeds up encode)

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special)

    # ------------------------------------------------------------------ training
    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        assert num_merges > 0, "vocab_size too small"

        # 1. Split into chunks and count each unique chunk.
        #    "the" may appear 50,000 times, but we only need to process it once.
        # (finditer reads chunks one at a time instead of building a huge list)
        chunk_counts = Counter(m.group() for m in self.pattern.finditer(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        # 2. Count every adjacent pair, and remember which words contain it.
        #    `where` lets us update only the words affected by each merge,
        #    instead of rescanning everything (this is what makes it fast).
        pair_counts = Counter()
        where = defaultdict(set)
        for wi, ids in enumerate(words):
            for p in zip(ids, ids[1:]):
                pair_counts[p] += freqs[wi]
                where[p].add(wi)

        # 3. Repeatedly merge the most frequent pair.
        for k in range(num_merges):
            if not pair_counts:
                break
            pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + k

            for wi in list(where[pair]):
                ids, f = words[wi], freqs[wi]
                # remove this word's old pairs from the counts...
                for p in zip(ids, ids[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    where[p].discard(wi)
                # ...merge, then add the word's new pairs back
                ids = merge(ids, pair, new_id)
                words[wi] = ids
                for p in zip(ids, ids[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
            where.pop(pair, None)

            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (k % 100 == 0 or k == num_merges - 1):
                print(f"merge {k + 1}/{num_merges}: {pair} -> {new_id} {self.vocab[new_id]!r}")

        # 4. Special tokens get the last ids.
        next_id = 256 + len(self.merges)
        self.special = {tok: next_id + i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.cache = {}

    # ------------------------------------------------------------------ encoding
    def _encode_chunk(self, chunk):
        if chunk in self.cache:
            return self.cache[chunk]
        ids = list(chunk.encode("utf-8"))
        # Apply merges in the order they were learned: always pick the pair
        # whose merge id is lowest (it was learned first).
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing left to merge
            ids = merge(ids, pair, self.merges[pair])
        # Cap the cache so memory doesn't grow forever on billions of tokens.
        # Common words get cached early, which is where the speedup comes from.
        if len(self.cache) < 500_000:
            self.cache[chunk] = ids
        return ids

    def encode(self, text, allow_special=True):
        # Pull special tokens out first so they aren't split into pieces.
        if allow_special and self.special:
            special_pattern = "(" + "|".join(re.escape(s) for s in self.special) + ")"
            parts = re.split(special_pattern, text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if part in self.special:
                ids.append(self.special[part])
            else:
                for chunk in self.pattern.findall(part):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    # ------------------------------------------------------------------ decoding
    def decode(self, ids):
        id_to_special = {i: tok for tok, i in self.special.items()}
        pieces = []
        for i in ids:
            if i in id_to_special:
                pieces.append(id_to_special[i].encode("utf-8"))
            else:
                pieces.append(self.vocab[i])
        # errors="replace": a half-finished emoji (incomplete bytes) won't crash
        return b"".join(pieces).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ save / load
    def save(self, path):
        data = {
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
            "special": self.special,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        for a, b, new_id in sorted(data["merges"], key=lambda m: m[2]):
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.special = data["special"]
        return tok


if __name__ == "__main__":
    import sys
    import time

    # Usage: python tokenizer.py input.txt 1024
    path = sys.argv[1]
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

    with open(path, encoding="utf-8") as f:
        text = f.read()

    tok = Tokenizer()
    t0 = time.time()
    tok.train(text, vocab_size, verbose=True)
    print(f"\ntrained in {time.time() - t0:.1f}s, vocab size = {tok.vocab_size}")
    tok.save("tokenizer.json")

    # Sanity checks
    sample = "<|user_start|>Hello there! How are you doing today?<|user_end|>"
    ids = tok.encode(sample)
    print(f"\n{sample!r}\n-> {ids}")
    print("pieces:", [tok.decode([i]) for i in ids])
    assert tok.decode(ids) == sample, "round trip failed!"

    reloaded = Tokenizer.load("tokenizer.json")
    assert reloaded.encode(sample) == ids, "save/load mismatch!"

    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tok.encode(text[:200_000]))
    print(f"compression: {len(text[:200_000].encode('utf-8')) / n_tokens:.2f} bytes per token")
    print("all checks passed")