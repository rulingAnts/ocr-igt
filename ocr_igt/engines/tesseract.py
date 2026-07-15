"""Offline Tesseract engine.

Tesseract is trained on clean *printed* text, so on sloppy low-literacy
handwriting it will do poorly — treat its output as a rough scaffold to correct
by hand, not a finished transcription. We keep the raw OCR lines in each phrase
so a human can fix things, and we flag every grouping/alignment guess.

Heuristic layout model (typical of these notebooks):

    <Fayu baseline line>
    <Indonesian gloss line, often run together>
    <Indonesian free translation>          (blank line separates examples)
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Document, Phrase, Word
from ..preprocess import preprocess_for_tesseract
from .base import Engine


class TesseractEngine(Engine):
    name = "tesseract"

    def recognize(self, image_path: str | Path) -> Document:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "The Tesseract engine needs pytesseract and the tesseract binary.\n"
                "  pip install pytesseract\n"
                "  brew install tesseract tesseract-lang   # macOS (incl. 'ind')"
            ) from exc

        img = preprocess_for_tesseract(image_path, do_pre=self.do_preprocess)
        lang = self._resolve_lang(pytesseract)

        try:
            text = pytesseract.image_to_string(img, lang=lang, config="--psm 6")
        except pytesseract.TesseractError as exc:  # pragma: no cover
            raise RuntimeError(f"Tesseract failed on {image_path}: {exc}") from exc

        doc = self._blank_document(image_path)
        doc.warnings.append(
            "Tesseract output on handwriting is unreliable — verify every line, "
            "the vern/gloss/free grouping, and the word↔gloss alignment."
        )
        doc.phrases = self._parse(text, doc.vernacular, doc.gloss_lang)
        if not doc.phrases:
            doc.warnings.append("No text recognized.")
        return doc

    def _resolve_lang(self, pytesseract) -> str:
        want = self.cfg.get("tesseract_lang", "ind")
        try:
            available = set(pytesseract.get_languages(config=""))
        except Exception:
            return want
        if want in available:
            return f"{want}+eng" if "eng" in available else want
        return "eng" if "eng" in available else (want or "eng")

    @staticmethod
    def _parse(text: str, vern: str, gloss: str) -> list[Phrase]:
        # Split into example blocks on blank lines, then map lines within a block
        # onto (baseline, gloss, free).
        blocks = re.split(r"\n[ \t]*\n", text.strip())
        phrases: list[Phrase] = []
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue
            phrases.append(TesseractEngine._block_to_phrase(lines))
        return phrases

    @staticmethod
    def _block_to_phrase(lines: list[str]) -> Phrase:
        baseline = lines[0]
        gloss_line = lines[1] if len(lines) > 1 else ""
        free = " ".join(lines[2:]) if len(lines) > 2 else ""

        vern_tokens = baseline.split()
        gloss_tokens = gloss_line.split()
        words: list[Word] = []
        for i, tok in enumerate(vern_tokens):
            g = gloss_tokens[i] if i < len(gloss_tokens) else ""
            words.append(Word(txt=tok, gloss=g))

        note_parts: list[str] = ["AUTO: verify line grouping"]
        if gloss_line and len(gloss_tokens) != len(vern_tokens):
            note_parts.append(
                f"word/gloss count mismatch ({len(vern_tokens)} words vs "
                f"{len(gloss_tokens)} glosses) — glosses may have run together"
            )
            leftover = gloss_tokens[len(vern_tokens):]
            if leftover:
                note_parts.append("unaligned glosses: " + " ".join(leftover))
        if not gloss_line:
            note_parts.append("no gloss line detected")

        return Phrase(
            words=words,
            free=free,
            note="; ".join(note_parts),
            raw_lines=lines,
        )
