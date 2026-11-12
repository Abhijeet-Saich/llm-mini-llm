"""
Step 5b: Pretrain on an H100 GPU on Modal.

Run from the mini-llm/ root. ALWAYS pass --detach (even for the speed test):
main() uses spawn(), so it returns immediately, and without --detach Modal
stops the app (and cancels the training) the moment main() returns.

1. Speed test (~5 min, about $0.50): measures speed and GPU memory.
    modal run --detach modal_app/train_modal.py --args "--preset rehearsal --max-steps 30 --log-every 5 --out-dir /vol/checkpoints/speedtest --compile"

2. Rehearsal (70M on ~1B tokens):
    modal run --detach modal_app/train_modal.py --args "--preset rehearsal --out-dir /vol/checkpoints/rehearsal --compile"

3. The real run (320M on ~6.5B tokens):
    modal run --detach modal_app/train_modal.py --args "--preset large --out-dir /vol/checkpoints/large --compile"

If a run stops for any reason, rerun the same command with --resume added
inside the quotes. It continues from the last saved checkpoint.
"""

import modal
import os

app = modal.App("mini-llm-train")
volume = modal.Volume.from_name("mini-llm")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "regex", "tqdm")
    .add_local_dir("minillm", remote_path="/root/minillm")
    .add_local_dir("scripts", remote_path="/root/scripts")
)


@app.function(
    image=image,
    gpu=os.environ.get("MINI_LLM_GPU", "H100"),   # pick the GPU from the command line
    volumes={"/vol": volume},
    memory=65536,                  # 64 GB RAM: the shards are copied to local disk
    timeout=24 * 60 * 60,          # Modal's maximum: 24 hours
)
def train(args: str):
    import shlex
    import shutil
    import subprocess
    import threading
    import time

    # 1. Copy the shards from the Volume to this machine's own disk. The Volume
    #    is a network drive; training reads thousands of small random windows
    #    per step, which is much faster from a local disk.
    t0 = time.time()
    shutil.copytree("/vol/data/fineweb", "/tmp/fineweb", dirs_exist_ok=True)
    print(f"copied shards to local disk in {time.time() - t0:.0f}s", flush=True)

    # 2. Every 5 minutes, save what's been written to /vol (checkpoints, log).
    #    If the machine dies suddenly, at most 5 minutes of saves are lost.
    stop = threading.Event()

    def commit_regularly():
        while not stop.wait(300):
            volume.commit()

    threading.Thread(target=commit_regularly, daemon=True).start()

    # 3. Run your own scripts/pretrain.py, unchanged, with the extra arguments.
    #    -u = print logs immediately instead of buffering them.
    cmd = [
        "python", "-u", "-m", "scripts.pretrain",
        "--data-dir", "/tmp/fineweb",
        "--tokenizer", "/vol/data/tokenizer.json",
    ] + shlex.split(args)
    print("running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd="/root", check=True)
    finally:
        stop.set()
        volume.commit()


@app.local_entrypoint()
def main(args: str = "--preset rehearsal --out-dir /vol/checkpoints/rehearsal --compile"):
    # spawn = start the training and DON'T wait for it. Requires `modal run --detach`:
    # without it, the app stops when this function returns and the run is cancelled.
    call = train.spawn(args)
    print(f"training started in the background (call id: {call.object_id})")
    print("if you used --detach, it's safe to close the laptop. "
          "Follow the logs on modal.com -> Apps -> mini-llm-train")