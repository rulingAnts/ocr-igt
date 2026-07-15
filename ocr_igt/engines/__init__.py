"""Pluggable OCR/structuring engines.

Each engine turns one image into a :class:`~ocr_igt.models.Document`. Tesseract
is fully offline; Claude uses the vision API. Select at runtime with ``--engine``.
"""

from __future__ import annotations

from typing import Any

from .base import Engine


def get_engine(name: str, cfg: dict[str, Any], do_preprocess: bool = True,
               do_dewarp: bool = True) -> Engine:
    name = name.lower()
    if name == "tesseract":
        from .tesseract import TesseractEngine
        return TesseractEngine(cfg, do_preprocess=do_preprocess, do_dewarp=do_dewarp)
    if name == "claude":
        from .claude import ClaudeEngine
        return ClaudeEngine(cfg, do_preprocess=do_preprocess, do_dewarp=do_dewarp)
    raise ValueError(f"unknown engine: {name!r} (choose 'tesseract' or 'claude')")


__all__ = ["Engine", "get_engine"]
