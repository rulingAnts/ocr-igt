"""Persisted config so the writing-system codes default to last-used values.

Stored at ``$XDG_CONFIG_HOME/ocr-igt/config.json`` (``~/.config/ocr-igt`` by
default). We never store the API key here — that stays in the environment.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "vernacular": "fau",        # Fayu baseline writing system (ISO 639-3: fau)
    "gloss_lang": "id",         # Indonesian glosses + free translations
    "analysis_lang": "en",      # segment numbers / analysis
    "engine": "tesseract",      # default OCR engine
    "model": "claude-opus-4-8", # used only by the claude engine
    "font": "Charis SIL",       # written into the .flextext <languages> block
    "tesseract_lang": "ind",    # Tesseract traineddata; falls back to eng
    "max_edge": 1568,           # longest image edge sent to the vision model
}


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "ocr-igt"


def config_path() -> Path:
    return config_dir() / "config.json"


def load_config() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    path = config_path()
    if path.exists():
        try:
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/unreadable config → fall back to defaults
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: cfg[k] for k in DEFAULTS if k in cfg}
    path.write_text(json.dumps(keep, indent=2) + "\n", encoding="utf-8")


def _ask(prompt: str, default: str) -> str:
    try:
        resp = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    return resp or default


def resolve_writing_systems(
    cfg: dict[str, Any],
    *,
    vern: str | None = None,
    gloss: str | None = None,
    analysis: str | None = None,
    interactive: bool = True,
) -> dict[str, Any]:
    """Return cfg updated with writing-system codes.

    Flags win. Otherwise prompt (defaulting to the saved value) when attached to
    a terminal; when non-interactive, silently keep the saved values.
    """

    if vern is not None:
        cfg["vernacular"] = vern
    if gloss is not None:
        cfg["gloss_lang"] = gloss
    if analysis is not None:
        cfg["analysis_lang"] = analysis

    need_prompt = interactive and sys.stdin.isatty() and (
        vern is None or gloss is None or analysis is None
    )
    if need_prompt:
        print("Writing-system codes for this FLEx project "
              "(press Enter to accept the saved default):")
        if vern is None:
            cfg["vernacular"] = _ask("  Vernacular (Fayu baseline)", cfg["vernacular"])
        if gloss is None:
            cfg["gloss_lang"] = _ask("  Gloss + free-translation (Indonesian)",
                                     cfg["gloss_lang"])
        if analysis is None:
            cfg["analysis_lang"] = _ask("  Analysis (segment numbers)",
                                        cfg["analysis_lang"])
    return cfg


# --------------------------------------------------------------------------- #
# Local spend ledger
#
# The Anthropic API does NOT expose your remaining prepaid balance (there is no
# public balance endpoint — check the Console for the real figure). As the next
# best thing, we tally what *this tool* has actually spent, appending one JSON
# line per run, so `ocr-igt` can show a running total before/after each batch.
# --------------------------------------------------------------------------- #
def spend_log_path() -> Path:
    return config_dir() / "spend.log"


def record_spend(model: str, pages: int, cost: float) -> None:
    path = spend_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "pages": pages,
        "cost_usd": round(cost, 6),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def total_spent() -> float:
    path = spend_log_path()
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            total += float(json.loads(line).get("cost_usd", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return total
