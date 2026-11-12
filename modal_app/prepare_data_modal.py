"""
Step 5a: Tokenize the full dataset on Modal and save it to a Modal Volume.

Run from the mini-llm/ root. ALWAYS pass --detach: main() uses spawn(), so it
returns immediately, and without --detach Modal stops the app (and cancels
the job) the moment main() returns.
    modal run --detach modal_app/prepare_data_modal.py

Quick test first (20M tokens, a few minutes, a few cents):
    modal run --detach modal_app/prepare_data_modal.py --max-tokens 20000000

This runs your own scripts/prepare_data.py, unchanged, on a 32-core machine.
The result lands on a Volume named "mini-llm": a disk that lives in your
Modal account and survives after the machine shuts down. Training (Step 5b)
will read the shards from that same Volume.
"""

import modal

app = modal.App("mini-llm-data")

# A Volume is persistent storage. create_if_missing makes it on the first run.
volume = modal.Volume.from_name("mini-llm", create_if_missing=True)

# The container image: Python plus the packages prepare_data.py needs, plus
# your own code and trained tokenizer copied in from your Mac.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "regex", "tqdm", "datasets")
    .add_local_dir("minillm", remote_path="/root/minillm")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_file("data/tokenizer.json", remote_path="/root/tokenizer.json")
)


@app.function(
    image=image,
    volumes={"/vol": volume},          # the Volume appears as the folder /vol
    cpu=32,                            # 32 cores = 32 tokenizer workers
    memory=32768,                      # 32 GB RAM
    timeout=12 * 60 * 60,               # stop after 6 hours no matter what
    secrets=[modal.Secret.from_name("huggingface")],
)
def prepare(max_tokens: int, val_tokens: int):
    import os
    import shutil
    import subprocess

    os.makedirs("/vol/data", exist_ok=True)
    # Keep the tokenizer next to the data it made: the two must always match.
    shutil.copy("/root/tokenizer.json", "/vol/data/tokenizer.json")

    subprocess.run(
        [
            "python", "-m", "scripts.prepare_data",
            "--tokenizer", "/vol/data/tokenizer.json",
            "--out-dir", "/vol/data/fineweb",
            "--max-tokens", str(max_tokens),
            "--val-tokens", str(val_tokens),
            "--workers", "32",
        ],
        cwd="/root",
        check=True,     # if prepare_data.py fails, fail loudly here too
    )
    volume.commit()     # make sure everything written to /vol is saved

    files = sorted(os.listdir("/vol/data/fineweb"))
    print(f"\n{len(files)} shard files on the Volume: {files[0]} ... {files[-1]}")


@app.local_entrypoint()
def main(max_tokens: int = 3_500_000_000, val_tokens: int = 10_000_000):
    # spawn = start the job and DON'T wait for it. Requires `modal run --detach`,
    # otherwise the app stops when this function returns and the job is cancelled.
    call = prepare.spawn(max_tokens, val_tokens)
    print(f"data prep started in the background (call id: {call.object_id})")
    print("if you used --detach, it's safe to close the laptop. "
          "Follow the logs on modal.com -> Apps -> mini-llm-data")