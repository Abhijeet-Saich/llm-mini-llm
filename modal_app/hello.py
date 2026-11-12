import modal

app = modal.App("mini-llm-hello")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")


@app.function(gpu="H100", image=image, timeout=300,
              secrets=[modal.Secret.from_name("huggingface")])
def hello():
    import os
    import torch
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("HF token found:", "HF_TOKEN" in os.environ)   # prints True/False, never the token


@app.local_entrypoint()
def main():
    hello.remote()