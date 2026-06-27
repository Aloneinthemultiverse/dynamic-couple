"""Model backend interface. Swap implementations (transformers / llama.cpp) freely.

Both Qwythos-9B (cuda:0) and Gemma 4 12B (cuda:1) implement this. Kaggle T4×2:
4-bit, fp16 compute, eager attention (Turing sm_75 has no bf16 / flash-attn 2).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass
class GenConfig:
    max_tokens: int = 1024
    temperature: float = 0.2
    stop: tuple[str, ...] = ()


class ModelHandle(Protocol):
    name: str
    device: str  # "cuda:0" | "cuda:1"

    def generate(self, prompt: str, cfg: GenConfig) -> str: ...

    def stream(self, prompt: str, cfg: GenConfig) -> Iterator[str]:
        """Token stream — feeds the direct pipeline (channel/pipeline.py)."""
        ...

    def healthy(self) -> bool:
        """In-process liveness. Feeds edit #1 (graph-only fallback)."""
        ...
