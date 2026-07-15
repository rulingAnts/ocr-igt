"""Engine interface shared by the Tesseract and Claude backends."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

from ..models import Document


class Engine(abc.ABC):
    name: str = "base"

    def __init__(self, cfg: dict[str, Any], do_preprocess: bool = True,
                 do_dewarp: bool = True) -> None:
        self.cfg = cfg
        self.do_preprocess = do_preprocess
        self.do_dewarp = do_dewarp

    @abc.abstractmethod
    def recognize(self, image_path: str | Path) -> Document:
        """Transcribe one image into an editable IGT Document."""
        raise NotImplementedError

    def _blank_document(self, image_path: str | Path) -> Document:
        return Document(
            source_image=str(image_path),
            engine=self.name,
            vernacular=self.cfg.get("vernacular", "fau"),
            gloss_lang=self.cfg.get("gloss_lang", "id"),
            analysis_lang=self.cfg.get("analysis_lang", "en"),
            title=Path(image_path).stem,
        )
