"""i18n at the EDGES only — detect input language, route to the better-fit model.
The plan->do->check loop stays language-neutral (code is code).
"""
from __future__ import annotations


def detect_language(text: str) -> str:
    raise NotImplementedError("langid/fasttext; default 'en'")


def route(language: str) -> str:
    """Return which model should field this language. Gemma = multimodal/multilingual,
    Qwythos = strong reasoning. Default: keep configured roles."""
    return "default"
