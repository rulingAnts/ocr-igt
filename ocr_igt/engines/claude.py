"""Claude vision engine — recommended for messy handwriting.

A vision LLM can be *told* the interlinear structure and return aligned
word↔gloss JSON in one shot, which is far more robust on low-literacy Fayu
handwriting than character-level OCR. Uses structured outputs so the response
is always schema-valid JSON.

Auth: reads ``ANTHROPIC_API_KEY`` from the environment (or an ``ant auth login``
profile). The key is never written to the config file. Every page prints its
token usage and estimated cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import Document, Phrase, Word
from ..preprocess import image_for_vision
from .base import Engine


# USD per 1M tokens (input, output). Update if Anthropic pricing changes.
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Rough token assumptions for the *pre-run* estimate (actuals are printed as the
# batch runs). Deliberately a slight over-estimate so the spend gate errs high.
_EST_PROMPT_TOKENS = 1100   # system + instructions
_EST_OUTPUT_TOKENS = 1400   # a page of IGT JSON (observed ~1100-1200)
_EST_IMG_TOKEN_CAP = 1600   # standard-res image token ceiling (~1568px)


def _estimate_image_tokens(path, max_edge: int) -> int:
    """Approximate the image tokens Claude will bill for one page."""
    try:
        import cv2
        import numpy as np

        arr = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return _EST_IMG_TOKEN_CAP
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > max_edge:
            scale = max_edge / float(longest)
            w, h = w * scale, h * scale
        return min(int(w * h / 750), _EST_IMG_TOKEN_CAP)
    except Exception:
        return _EST_IMG_TOKEN_CAP


def estimate_cost(image_paths, model: str, max_edge: int) -> tuple[int, int, float]:
    """Return (est_input_tokens, est_output_tokens, est_cost_usd) for a batch."""
    pin, pout = _PRICING.get(model, (5.0, 25.0))
    total_in = total_out = 0
    for p in image_paths:
        total_in += _EST_PROMPT_TOKENS + _estimate_image_tokens(p, max_edge)
        total_out += _EST_OUTPUT_TOKENS
    cost = total_in / 1e6 * pin + total_out / 1e6 * pout
    return total_in, total_out, cost

_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "phrases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "words": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "txt": {"type": "string"},
                                "gloss": {"type": "string"},
                            },
                            "required": ["txt", "gloss"],
                            "additionalProperties": False,
                        },
                    },
                    "free": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["words", "free", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "phrases"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are an expert field linguist transcribing interlinear glossed text (IGT) "
    "from handwritten notebook scans of the Fayu language (ISO 639-3: fau), a "
    "Papuan language of Indonesia. Glosses and free translations are in Indonesian "
    "and are frequently misspelled by low-literacy writers. You transcribe exactly "
    "what is written — you never correct, normalize, or invent."
)

_INSTRUCTIONS = """\
This image is one page of a Fayu field notebook. It contains interlinear glossed
examples. Each example is typically three lines:

  1. a FAYU baseline line (the words of the sentence);
  2. an INDONESIAN gloss line, word-by-word under the baseline — but the glosses
     are often written run together, cramped, or without clear spacing;
  3. an INDONESIAN free translation of the whole sentence (often between examples).

Produce structured IGT with these rules:

- Transcribe the Fayu baseline EXACTLY as written, including apparent misspellings
  and non-standard letters. Do not standardize.
- Split each example into `words`. For every Fayu baseline word, give its `txt`
  (the Fayu word) and `gloss` (the Indonesian gloss written beneath/near it).
- The gloss line may run together. Segment it and align each gloss to its baseline
  word as best you can. If you cannot confidently align a gloss to a word, leave
  that word's `gloss` empty and describe the problem in the phrase `note`.
- `free` is the Indonesian free translation of that example. If none is present,
  leave it empty.
- Preserve Indonesian misspellings in glosses and free translations — do NOT fix them.
- If a Fayu word is unreadable, use "???" for its `txt`. If a whole line is
  illegible, add a phrase with a `note` explaining that.
- Never invent content that is not visibly on the page. Empty is better than guessed.
- Use `note` for any uncertainty (unclear handwriting, ambiguous grouping, etc.).
- `title`: a short label for the page (e.g. a page number or heading if visible),
  otherwise "".

Return one entry in `phrases` per interlinear example, in top-to-bottom order.
"""


class ClaudeEngine(Engine):
    name = "claude"

    def __init__(self, cfg: dict[str, Any], do_preprocess: bool = True) -> None:
        super().__init__(cfg, do_preprocess=do_preprocess)
        self.model = cfg.get("model", "claude-opus-4-8")
        self.max_edge = int(cfg.get("max_edge", 1568))
        self._client = None
        # Running totals, surfaced by the CLI after a batch.
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "The Claude engine needs the anthropic SDK:\n"
                    "  pip install anthropic"
                ) from exc
            try:
                self._client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Could not initialize the Anthropic client. Set ANTHROPIC_API_KEY "
                    "or run `ant auth login`.\n"
                    f"  ({exc})"
                ) from exc
        return self._client

    def recognize(self, image_path: str | Path) -> Document:
        import json

        client = self._get_client()
        b64, media_type = image_for_vision(
            image_path, do_pre=self.do_preprocess, max_edge=self.max_edge
        )

        resp = client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": _INSTRUCTIONS},
                    ],
                }
            ],
        )

        self._account(resp.usage)

        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - schema should prevent
            doc = self._blank_document(image_path)
            doc.warnings.append(f"Model returned unparseable JSON: {exc}")
            return doc

        return self._to_document(image_path, data, resp)

    def _to_document(self, image_path, data: dict, resp) -> Document:
        doc = self._blank_document(image_path)
        doc.title = str(data.get("title") or Path(image_path).stem)
        for p in data.get("phrases", []):
            words = [
                Word(txt=str(w.get("txt", "")), gloss=str(w.get("gloss", "")))
                for w in p.get("words", [])
            ]
            doc.phrases.append(
                Phrase(
                    words=words,
                    free=str(p.get("free", "")),
                    note=str(p.get("note", "")),
                )
            )
        if getattr(resp, "stop_reason", None) == "refusal":
            doc.warnings.append("Model refused to process this image.")
        if not doc.phrases:
            doc.warnings.append("Model returned no phrases for this page.")
        return doc

    def _account(self, usage) -> None:
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        pin, pout = _PRICING.get(self.model, (5.0, 25.0))
        cost = in_tok / 1e6 * pin + out_tok / 1e6 * pout
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self.total_cost += cost
        self.last_cost = cost

    def cost_summary(self) -> str:
        return (
            f"{self.total_input_tokens:,} input + {self.total_output_tokens:,} output "
            f"tokens  ≈  ${self.total_cost:.3f} ({self.model})"
        )
