"""The intermediate IGT data model and its JSON sidecar (de)serialization.

The sidecar is deliberately simple and human-editable — a linguist opens the
`*.igt.json`, fixes the messy OCR, then runs `build`. Word glossing is at the
*word* level (one Indonesian gloss per Fayu word), which is what a handwritten
gloss line gives you and what FLEx imports cleanly as a word-gloss line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SIDECAR_SUFFIX = ".igt.json"


@dataclass
class Word:
    txt: str = ""            # Fayu baseline word, transcribed as written
    gloss: str = ""          # Indonesian gloss (may be misspelled, that's fine)

    def to_dict(self) -> dict[str, str]:
        return {"txt": self.txt, "gloss": self.gloss}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Word":
        return cls(txt=str(d.get("txt", "")), gloss=str(d.get("gloss", "")))


@dataclass
class Phrase:
    words: list[Word] = field(default_factory=list)
    free: str = ""           # Indonesian free translation of the whole line
    note: str = ""           # reviewer/OCR uncertainty note (not exported unless kept)
    raw_lines: list[str] = field(default_factory=list)  # raw OCR text, for reference

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "words": [w.to_dict() for w in self.words],
            "free": self.free,
            "note": self.note,
        }
        if self.raw_lines:
            d["raw_lines"] = self.raw_lines
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Phrase":
        return cls(
            words=[Word.from_dict(w) for w in d.get("words", [])],
            free=str(d.get("free", "")),
            note=str(d.get("note", "")),
            raw_lines=[str(x) for x in d.get("raw_lines", [])],
        )


@dataclass
class Document:
    """One transcribed notebook page → one FLEx interlinear-text."""

    source_image: str = ""
    engine: str = ""
    vernacular: str = "fau"        # writing-system code for the Fayu baseline
    gloss_lang: str = "id"         # writing-system code for glosses + free trans.
    analysis_lang: str = "en"      # writing-system code for segment numbers
    title: str = ""
    phrases: list[Phrase] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_image": self.source_image,
            "engine": self.engine,
            "vernacular": self.vernacular,
            "gloss_lang": self.gloss_lang,
            "analysis_lang": self.analysis_lang,
            "title": self.title,
            "phrases": [p.to_dict() for p in self.phrases],
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        return cls(
            source_image=str(d.get("source_image", "")),
            engine=str(d.get("engine", "")),
            vernacular=str(d.get("vernacular", "fau")),
            gloss_lang=str(d.get("gloss_lang", "id")),
            analysis_lang=str(d.get("analysis_lang", "en")),
            title=str(d.get("title", "")),
            phrases=[Phrase.from_dict(p) for p in d.get("phrases", [])],
            warnings=[str(x) for x in d.get("warnings", [])],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        header = (
            "// Edit this file to correct the OCR, then run `ocr-igt build`.\n"
            "// One phrase = one glossed line; each word has a Fayu `txt` and an\n"
            "// Indonesian `gloss`. `free` is the Indonesian free translation.\n"
            "// `raw_lines`/`note`/`warnings` are for your reference only.\n"
        )
        with path.open("w", encoding="utf-8") as fh:
            fh.write(header)
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "Document":
        text = Path(path).read_text(encoding="utf-8")
        # Tolerate the `//` comment header we write above.
        cleaned = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("//")
        )
        return cls.from_dict(json.loads(cleaned))
