"""
Steps 5-6: run training scripts on a Modal GPU.

Always launch with --detach. The run is started with .spawn(), so it belongs
to Modal, not to your terminal: closing the laptop can't cancel it.
Pick the GPU with MINI_LLM_GPU (default H100), e.g. MINI_LLM_GPU=A100-80GB.

Pretraining (Step 5):
    modal run --detach modal_app/train_modal.py --args "--preset large --out-dir /vol/checkpoints/large --compile"

Chat tuning (Step 6):
    modal run --detach modal_app/train_modal.py --script scripts.sft --args "--init /vol/checkpoints/rehearsal/ckpt.pt --out-dir /vol/checkpoints/rehearsal-sft"

Watch the logs:  modal app logs <app ID>
"""

import os

import modal

app = modal.App("mini-llm-train")
volume = modal.Volume.from_name("mini-llm")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "regex", "tqdm", "datasets")
    .add_local_dir("minillm", remote_path="/root/minillm")
    .add_local_dir("scripts", remote_path="/root/scripts")
)


@app.function(
    image=image,
    gpu=os.environ.get("MINI_LLM_GPU", "H100"),   # pick the GPU from the command line
    cpu=8,                                         # for data loading and tokenizing (SFT)
    volumes={"/vol": volume},
    memory=65536,                                  # 64 GB RAM
    timeout=24 * 60 * 60,                          # Modal's maximum: 24 hours
    secrets=[modal.Secret.from_name("huggingface")],   # SFT downloads its dataset
)
def train(args: str, script: str = "scripts.pretrain"):
    import shlex
    import shutil
    import subprocess
    import threading
    import time

    extra = ["--tokenizer", "/vol/data/tokenizer.json"]
    if script == "scripts.pretrain":
        # Pretraining reads thousands of random windows per step: much faster
        # from this machine's own disk than from the network Volume.
        t0 = time.time()
        shutil.copytree("/vol/data/fineweb", "/tmp/fineweb", dirs_exist_ok=True)
        print(f"copied shards to local disk in {time.time() - t0:.0f}s", flush=True)
        extra += ["--data-dir", "/tmp/fineweb"]

    # Every 5 minutes, save what's been written to /vol (checkpoints, logs).
    stop = threading.Event()

    def commit_regularly():
        while not stop.wait(300):
            volume.commit()

    threading.Thread(target=commit_regularly, daemon=True).start()

    cmd = ["python", "-u", "-m", script] + extra + shlex.split(args)
    print("running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd="/root", check=True)
    finally:
        stop.set()
        volume.commit()


@app.local_entrypoint()
def main(args: str = "--preset rehearsal --out-dir /vol/checkpoints/rehearsal --compile",
         script: str = "scripts.pretrain"):
    # spawn = start the run and DON'T wait for it, so nothing on your laptop
    # is connected to it afterwards.
    call = train.spawn(args, script)
    print(f"started {script} in the background (call id: {call.object_id})")
    print("if you used --detach, it's safe to close the laptop. Logs: modal app logs <app ID>")