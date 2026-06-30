"""Kaggle T4×2 in-process loader — GGUF + llama.cpp (NOT transformers/bitsandbytes).

Proven by the qwythos-solo kernel: models are GGUF, downloaded from HuggingFace at runtime
(enable_internet), loaded with llama-cpp-python on GPU. One model per T4.
  Qwythos-9B -> GPU0  (empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF) — already proven
  Gemma 4 12B -> GPU1 (needs a GGUF HF repo id — same download path, no Kaggle upload)
T4 = Turing sm_75: GGUF/llama.cpp sidesteps the bf16 / flash-attn 2 limits entirely.
"""
from __future__ import annotations
import os

QWY_REPO = os.environ.get("QWY_REPO", "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF")
GEMMA_REPO = os.environ.get("GEMMA_REPO", "ggml-org/gemma-4-12B-it-GGUF")


def _pick_gguf(repo: str, prefer: str = "q4"):
    from huggingface_hub import hf_hub_download, list_repo_files
    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    pref = [f for f in files if prefer in f.lower()]
    pick = (pref or files)[0]
    return hf_hub_download(repo, pick)


def load(repo: str, main_gpu: int, n_ctx: int = 8192):
    """Download a GGUF from HF and load it on the given T4 via llama.cpp."""
    from llama_cpp import Llama
    path = _pick_gguf(repo)
    return Llama(model_path=path, n_gpu_layers=-1, n_ctx=n_ctx,
                 main_gpu=main_gpu, verbose=False)
