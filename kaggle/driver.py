# dynamic-couple | Kaggle T4x2 runtime SMOKE TEST
# Validates the runtime before the full couple loop: GPU visibility, GGUF download,
# llama.cpp load on GPU, one short generation. Mirrors the proven qwythos-solo setup.
# Qwythos = GGUF from HuggingFace (NOT a Kaggle mount); Gemma will load the same way.
import os, subprocess, sys

def sh(c): subprocess.run(c, shell=True, check=True)

print("=== GPU visibility ===", flush=True)
sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv || true")

# llama.cpp python bindings with CUDA (T4 = sm_75; no flash-attn 2 needed for GGUF)
print("\n=== install llama-cpp-python (CUDA) ===", flush=True)
os.environ["CMAKE_ARGS"] = "-DGGML_CUDA=on"
sh("pip -q install huggingface_hub llama-cpp-python || pip -q install huggingface_hub llama-cpp-python")

from huggingface_hub import hf_hub_download, list_repo_files
QWY_REPO = os.environ.get("QWY_REPO", "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF")

def pick_gguf(repo):
    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    # prefer a Q4 quant to fit a single T4 (16GB)
    q4 = [f for f in files if "q4" in f.lower()]
    pick = (q4 or files)[0]
    print("  ", repo, "->", pick, flush=True)
    return hf_hub_download(repo, pick)

print("\n=== download Qwythos GGUF ===", flush=True)
path = pick_gguf(QWY_REPO)

print("\n=== load on GPU0 + smoke generation ===", flush=True)
from llama_cpp import Llama
llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=4096, main_gpu=0, verbose=False)
out = llm("Write a Python function that returns the nth Fibonacci number.\n", max_tokens=128)
print(out["choices"][0]["text"], flush=True)
print("\n*** SMOKE TEST OK — Qwythos GGUF loads + generates on T4. "
      "Next: add Gemma on GPU1 + wire couple loop. ***", flush=True)
