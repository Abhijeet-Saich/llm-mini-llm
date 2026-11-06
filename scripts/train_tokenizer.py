"""
Step 2a: Train the real tokenizer on FineWeb-Edu web text.

Run from the mini-llm/ root folder:
    python -m scripts.train_tokenizer
"""

import argparse
import time

from minillm.fineweb import iter_documents
from minillm.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--train-chars", type=int, default=20_000_000, help="how much text to learn merges from")
    parser.add_argument("--test-chars", type=int, default=1_000_000, help="extra text, NOT used for training, to measure compression")
    parser.add_argument("--out", default="data/tokenizer.json")
    args = parser.parse_args()

    # 1. Collect documents. The first `train_chars` characters are for training,
    #    the next `test_chars` are held out so we test on text the tokenizer never saw.
    print("downloading documents...")
    train_docs, test_docs = [], []
    n_chars = 0
    for text in iter_documents():
        if n_chars < args.train_chars:
            train_docs.append(text)
        else:
            test_docs.append(text)
        n_chars += len(text)
        if n_chars >= args.train_chars + args.test_chars:
            break
    print(f"got {len(train_docs)} training docs, {len(test_docs)} test docs")

    # 2. Train. Joining with "\n" is fine: merges never cross chunk borders anyway.
    tok = Tokenizer()
    t0 = time.time()
    tok.train("\n".join(train_docs), args.vocab_size, verbose=True)
    print(f"\ntrained in {(time.time() - t0) / 60:.1f} min, vocab size = {tok.vocab_size}")
    tok.save(args.out)
    print(f"saved to {args.out}")

    # 3. Report. Compression = bytes of text per token; higher means fewer tokens.
    test_text = "\n".join(test_docs)
    ids = tok.encode(test_text, allow_special=False)
    assert tok.decode(ids) == test_text, "round trip failed!"
    print(f"\ncompression on unseen text: {len(test_text.encode('utf-8')) / len(ids):.2f} bytes per token")

    print("\nlast 20 tokens learned (the rarest merges):")
    last_ids = sorted(tok.merges.values())[-20:]
    print([tok.decode([i]) for i in last_ids])

    sample = test_docs[0][:200]
    print(f"\nexample: {sample!r}")
    print([tok.decode([i]) for i in tok.encode(sample, allow_special=False)])


if __name__ == "__main__":
    main()