"""Kaggle T4×2 in-process loader. 4-bit, fp16 compute, eager attention (Turing sm_75).

One model per GPU. Weights mounted read-only from Kaggle Models at /kaggle/input/.
  Qwythos-9B  -> cuda:0   (already on Kaggle)
  Gemma 4 12B -> cuda:1   (MUST be pushed as a Kaggle Model first)
"""
from __future__ import annotations

QWYTHOS_PATH = "/kaggle/input/qwythos-9b"      # confirm exact dataset slug
GEMMA_PATH = "/kaggle/input/gemma-4-12b"        # TODO: push this Model to Kaggle


def load_4bit(path: str, device: str):
    """Load a model in 4-bit on the given T4. fp16 compute, attn_implementation='eager'."""
    raise NotImplementedError(
        "transformers BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=fp16); "
        "device_map={'': device}; attn_implementation='eager'  (T4 = no flash-attn 2)"
    )
