# dynamic-couple | Kaggle T4x2 runtime SMOKE TEST (both models)
# Validates the runtime before the full couple loop: GPU visibility, GGUF download,
# llama.cpp load of BOTH models (one per T4), one short generation each.
# GGUF from HuggingFace at runtime (NOT Kaggle mounts). Qwythos=DOER, Gemma=PLANNER.
import os, subprocess

def sh(c): subprocess.run(c, shell=True, check=True)

print("=== GPU visibility ===", flush=True)
sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv || true")

print("\n=== install llama-cpp-python (prebuilt CUDA 12.4 wheel — no source build) ===", flush=True)
sh("pip -q install huggingface_hub")
sh("pip -q install 'llama-cpp-python[server]' "
   "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")

from huggingface_hub import hf_hub_download, list_repo_files
from llama_cpp import Llama

QWY_REPO = os.environ.get("QWY_REPO", "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF")
GEMMA_REPO = os.environ.get("GEMMA_REPO", "ggml-org/gemma-4-12B-it-GGUF")

def pick_gguf(repo, prefer="q4"):
    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    pref = [f for f in files if prefer in f.lower()]
    pick = (pref or files)[0]
    print("  ", repo, "->", pick, flush=True)
    return hf_hub_download(repo, pick)

def load_and_probe(repo, gpu, prompt):
    print(f"\n=== load {repo} on GPU{gpu} ===", flush=True)
    path = pick_gguf(repo)
    llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=4096, main_gpu=gpu, verbose=False)
    out = llm(prompt, max_tokens=128)
    print(out["choices"][0]["text"], flush=True)
    return llm

# DOER on GPU0, PLANNER on GPU1 — proves both T4s independently
load_and_probe(QWY_REPO, 0, "Write a Python function that returns the nth Fibonacci number.\n")
load_and_probe(GEMMA_REPO, 1, "Break this task into 3 numbered steps: add a CSV export endpoint.\n")

print("\n*** SMOKE TEST OK — both GGUF models load + generate on T4x2. "
      "Next: wire couple loop (controller.py) + SWE-bench in-process runner. ***", flush=True)
