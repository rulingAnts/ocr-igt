"""Emit a FLEx-importable ``.flextext`` (document version 2) from Documents.

Structure produced (one <interlinear-text> per source page):

    <document version="2">
      <interlinear-text>
        <item type="title" lang="id">...</item>
        <paragraphs><paragraph><phrases>
          <phrase>
            <item type="segnum" lang="en">1</item>
            <words>
              <word><item type="txt" lang="fau">abogo</item>
                    <item type="gls" lang="id">pergi</item></word>
              <word><item type="punct" lang="fau">.</item></word>
            </words>
            <item type="gls" lang="id">Dia pergi.</item>   (free translation)
          </phrase>
        </phrases></paragraph></paragraphs>
        <languages>...</languages>
      </interlinear-text>
    </document>
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .models import Document


# Characters split off a baseline word into their own <item type="punct"> word.
_PUNCT = set(".,!?;:…\"'()[]{}—–“”‘’")


def _split_word_token(token: str) -> list[tuple[str, bool]]:
    """Peel leading/trailing punctuation off a token into separate pieces.

    Only edge punctuation is split — internal marks (hyphens, apostrophes in
    Fayu words) are kept. Returns ordered (text, is_punct) pieces.

        "pergi."  -> [("pergi", False), (".", True)]
        "abogo"   -> [("abogo", False)]
        "???"     -> [("???", False)]   # illegible-word placeholder, not punct
        "."       -> [(".", True)]
    """
    start, end = 0, len(token)
    while start < end and token[start] in _PUNCT:
        start += 1
    while end > start and token[end - 1] in _PUNCT:
        end -= 1
    lead, core, trail = token[:start], token[start:end], token[end:]

    if not core:
        # Entire token is punctuation. Treat a run of '?' as an illegible
        # baseline-word placeholder; anything else is genuine punctuation.
        if len(token) >= 2 and set(token) == {"?"}:
            return [(token, False)]
        return [(token, True)]

    pieces: list[tuple[str, bool]] = []
    if lead:
        pieces.append((lead, True))
    pieces.append((core, False))
    if trail:
        pieces.append((trail, True))
    return pieces


def _item(parent: ET.Element, typ: str, lang: str, text: str) -> None:
    el = ET.SubElement(parent, "item", {"type": typ, "lang": lang})
    el.text = text


def _add_interlinear_text(root: ET.Element, doc: Document) -> None:
    it = ET.SubElement(root, "interlinear-text")
    title = doc.title or Path(doc.source_image).stem or "Untitled"
    _item(it, "title", doc.gloss_lang, title)

    paragraphs = ET.SubElement(it, "paragraphs")
    paragraph = ET.SubElement(paragraphs, "paragraph")
    phrases_el = ET.SubElement(paragraph, "phrases")

    for i, phrase in enumerate(doc.phrases, start=1):
        ph = ET.SubElement(phrases_el, "phrase")
        _item(ph, "segnum", doc.analysis_lang, str(i))
        words_el = ET.SubElement(ph, "words")

        for word in phrase.words:
            token = (word.txt or "").strip()
            gloss = (word.gloss or "").strip()
            if not token:
                if gloss:  # gloss with no baseline word — keep it visible
                    w = ET.SubElement(words_el, "word")
                    _item(w, "txt", doc.vernacular, "∅")
                    _item(w, "gls", doc.gloss_lang, gloss)
                continue

            pieces = _split_word_token(token)
            gloss_used = False
            for text, is_punct in pieces:
                w = ET.SubElement(words_el, "word")
                if is_punct:
                    _item(w, "punct", doc.vernacular, text)
                else:
                    _item(w, "txt", doc.vernacular, text)
                    # Attach the gloss to the first non-punct piece only.
                    if gloss and not gloss_used:
                        _item(w, "gls", doc.gloss_lang, gloss)
                        gloss_used = True

        if phrase.free.strip():
            _item(ph, "gls", doc.gloss_lang, phrase.free.strip())
        if phrase.note.strip():
            _item(ph, "note", doc.gloss_lang, phrase.note.strip())


def build_flextext(docs: list[Document], font: str = "Charis SIL") -> str:
    """Return the full .flextext XML string for a list of pages."""
    root = ET.Element("document", {"version": "2"})
    for doc in docs:
        _add_interlinear_text(root, doc)
    # FLEx puts one <languages> block inside each <interlinear-text>.
    _distribute_languages(root, docs, font)
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode")
    return "<?xml version='1.0' encoding='utf-8'?>\n" + xml + "\n"


def _distribute_languages(root: ET.Element, docs: list[Document], font: str) -> None:
    for it, doc in zip(root.findall("interlinear-text"), docs):
        langs = ET.SubElement(it, "languages")
        seen: set[str] = set()
        for code, vern in (
            (doc.vernacular, True),
            (doc.gloss_lang, False),
            (doc.analysis_lang, False),
        ):
            if code and code not in seen:
                seen.add(code)
                attrs = {"lang": code, "font": font}
                if vern:
                    attrs["vernacular"] = "true"
                ET.SubElement(langs, "language", attrs)


def write_flextext(docs: list[Document], out_path: str | Path,
                   font: str = "Charis SIL") -> None:
    Path(out_path).write_text(build_flextext(docs, font=font), encoding="utf-8")
