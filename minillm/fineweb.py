"""
Streams documents from FineWeb-Edu: high-quality educational web pages,
filtered by Hugging Face. "Streaming" means we read documents one by one over
the internet instead of downloading the whole 10-billion-token sample first.
"""

from datasets import load_dataset


def iter_documents():
    """Yield one document (a string of text) at a time, forever in the same order."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",   # a 10-billion-token random sample; plenty for us
        split="train",
        streaming=True,
    )
    for row in ds:
        yield row["text"]