"""Command-line interface: `ocr-igt ocr | build | config`."""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path
from typing import Iterator

from . import __version__
from .config import (
    DEFAULTS,
    config_path,
    load_config,
    record_spend,
    resolve_writing_systems,
    save_config,
    total_spent,
)
from .models import SIDECAR_SUFFIX, Document

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}


# --------------------------------------------------------------------------- #
# Input expansion
# --------------------------------------------------------------------------- #
def _iter_images_in(path: Path) -> Iterator[Path]:
    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                yield p
    elif path.suffix.lower() in IMAGE_EXTS:
        yield path


def _render_pdf(pdf: Path, out_dir: Path, dpi: int = 300) -> list[Path]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            f"{pdf.name} is a PDF — install PyMuPDF to render it, or convert it to "
            "images first:\n  pip install PyMuPDF"
        ) from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    doc = fitz.open(pdf)
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        out = out_dir / f"{pdf.stem}_p{i:03d}.png"
        pix.save(out)
        rendered.append(out)
    doc.close()
    return rendered


def expand_ocr_inputs(inputs: list[str], out_dir: Path, dpi: int) -> list[Path]:
    """Turn files/dirs/PDFs into a flat list of concrete image paths."""
    images: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if not path.exists():
            print(f"warning: no such path: {path}", file=sys.stderr)
            continue
        if path.is_dir():
            images.extend(_iter_images_in(path))
            for pdf in sorted(path.glob("*.pdf")):
                images.extend(_render_pdf(pdf, out_dir, dpi))
        elif path.suffix.lower() in PDF_EXTS:
            images.extend(_render_pdf(path, out_dir, dpi))
        elif path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
        else:
            print(f"warning: skipping unsupported file: {path}", file=sys.stderr)
    return images


def _apply_excludes(images: list[Path], patterns: list[str] | None
                    ) -> tuple[list[Path], list[Path]]:
    """Drop images whose filename matches any --exclude pattern.

    A pattern matches by exact filename, by stem, or as a shell glob — so
    ``--exclude tosokai_0.jpg``, ``--exclude tosokai_0``, and
    ``--exclude 'tosokai_0.*'`` all work.
    """
    if not patterns:
        return images, []
    kept: list[Path] = []
    skipped: list[Path] = []
    for img in images:
        if any(
            img.name == pat or img.stem == pat or fnmatch.fnmatch(img.name, pat)
            for pat in patterns
        ):
            skipped.append(img)
        else:
            kept.append(img)
    return kept, skipped


def _iter_sidecars(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob(f"*{SIDECAR_SUFFIX}")))
        elif path.exists():
            files.append(path)
        else:
            print(f"warning: no such path: {path}", file=sys.stderr)
    return files


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_ocr(args: argparse.Namespace) -> int:
    cfg = load_config()
    cfg["engine"] = args.engine or cfg.get("engine", "tesseract")
    if args.model:
        cfg["model"] = args.model

    cfg = resolve_writing_systems(
        cfg,
        vern=args.vern,
        gloss=args.gloss,
        analysis=args.analysis,
        interactive=not args.yes,
    )
    save_config(cfg)  # remember these codes for next time

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    render_dir = out_dir or Path.cwd()

    images = expand_ocr_inputs(args.inputs, render_dir, args.dpi)
    images, skipped = _apply_excludes(images, args.exclude)
    for s in skipped:
        print(f"Excluded: {s.name}")
    if not images:
        print("No images to process.", file=sys.stderr)
        return 1

    from .engines import get_engine

    try:
        engine = get_engine(cfg["engine"], cfg,
                            do_preprocess=not args.no_preprocess,
                            do_dewarp=not args.no_dewarp)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Engine: {cfg['engine']}"
          + (f" ({cfg['model']})" if cfg["engine"] == "claude" else "")
          + f"  |  vernacular={cfg['vernacular']} gloss={cfg['gloss_lang']}")

    if cfg["engine"] == "claude":
        from .engines.claude import estimate_cost

        est_in, est_out, est_cost = estimate_cost(
            images, cfg["model"], int(cfg["max_edge"])
        )
        print(f"\n{len(images)} page(s) will be sent to the Anthropic API "
              f"({cfg['model']}).")
        print(f"  Estimated cost:  ~${est_cost:.2f}   "
              f"(~{est_in:,} input + ~{est_out:,} output tokens; rough, usually high)")
        print(f"  Spent by ocr-igt so far (local tally): ${total_spent():.2f}")
        print("  Live account balance isn't available via the API — check "
              "https://console.anthropic.com/settings/billing")
        if not args.yes:
            if not sys.stdin.isatty():
                print("\nNon-interactive shell — re-run with --yes to proceed.",
                      file=sys.stderr)
                return 2
            try:
                resp = input("Proceed? [y/N]: ").strip().lower()
            except EOFError:
                resp = ""
            if resp not in ("y", "yes"):
                print("Aborted — nothing sent, no cost incurred.")
                return 0
        print()

    written = 0
    for image in images:
        try:
            doc = engine.recognize(image)
        except (RuntimeError, ValueError) as exc:
            print(f"  {image.name}: FAILED — {exc}", file=sys.stderr)
            if isinstance(exc, RuntimeError) and cfg["engine"] == "claude":
                return 1  # auth/SDK problems won't fix themselves; stop early
            continue

        sidecar = _sidecar_path(image, out_dir)
        doc.save(sidecar)
        written += 1
        extra = ""
        if cfg["engine"] == "claude" and hasattr(engine, "last_cost"):
            extra = f"  (${engine.last_cost:.3f})"
        print(f"  {image.name} → {sidecar.name}"
              f"  [{len(doc.phrases)} phrases]{extra}")

    print(f"\nWrote {written} sidecar(s).")
    if cfg["engine"] == "claude" and hasattr(engine, "cost_summary"):
        if getattr(engine, "total_cost", 0) > 0:
            record_spend(cfg["model"], written, engine.total_cost)
        print("Total this run: " + engine.cost_summary())
        print(f"Spent by ocr-igt to date (local tally): ${total_spent():.2f}")
    print("Review/correct the .igt.json files, then run: ocr-igt build <dir>")
    return 0 if written else 1


def _sidecar_path(image: Path, out_dir: Path | None) -> Path:
    name = image.stem + SIDECAR_SUFFIX
    return (out_dir / name) if out_dir else image.with_name(name)


def cmd_build(args: argparse.Namespace) -> int:
    cfg = load_config()
    files = _iter_sidecars(args.inputs)
    if not files:
        print("No .igt.json sidecars found.", file=sys.stderr)
        return 1

    docs: list[Document] = []
    for f in files:
        try:
            docs.append(Document.load(f))
        except Exception as exc:
            print(f"warning: skipping {f}: {exc}", file=sys.stderr)
    if not docs:
        print("Nothing to build.", file=sys.stderr)
        return 1

    from .flextext import write_flextext

    out = Path(args.out)
    font = args.font or cfg.get("font", "Charis SIL")
    write_flextext(docs, out, font=font)

    total_phrases = sum(len(d.phrases) for d in docs)
    print(f"Wrote {out}  ({len(docs)} text(s), {total_phrases} phrases, font={font!r})")
    print("Import into FLEx via: File ▸ Import ▸ FLExText Interlinear.")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config()
    if args.set:
        for pair in args.set:
            if "=" not in pair:
                print(f"error: --set expects key=value, got {pair!r}", file=sys.stderr)
                return 1
            key, val = pair.split("=", 1)
            if key not in DEFAULTS:
                print(f"error: unknown key {key!r}. Known: {', '.join(DEFAULTS)}",
                      file=sys.stderr)
                return 1
            cfg[key] = int(val) if isinstance(DEFAULTS[key], int) else val
        save_config(cfg)
        print(f"Saved {config_path()}")

    print(f"Config file: {config_path()}")
    for key in DEFAULTS:
        print(f"  {key} = {cfg.get(key)!r}")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ocr-igt",
        description="OCR handwritten Fayu interlinear notebooks into FLEx .flextext.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # ocr
    o = sub.add_parser("ocr", help="transcribe images/PDFs into editable .igt.json")
    o.add_argument("inputs", nargs="+", help="image files, folders, or PDFs")
    o.add_argument("--engine", choices=["tesseract", "claude"],
                   help="OCR engine (default: saved value, initially tesseract)")
    o.add_argument("--exclude", action="append", metavar="NAME|GLOB",
                   help="skip files by name, stem, or glob (repeatable), "
                        "e.g. --exclude tosokai_0.jpg")
    o.add_argument("--out", metavar="DIR", help="output dir for sidecars "
                   "(default: alongside each image)")
    o.add_argument("--vern", help="vernacular (Fayu) writing-system code")
    o.add_argument("--gloss", help="gloss/free-translation (Indonesian) code")
    o.add_argument("--analysis", help="analysis (segnum) writing-system code")
    o.add_argument("--model", help="Claude model id (claude engine only)")
    o.add_argument("--dpi", type=int, default=300, help="PDF render DPI (default 300)")
    o.add_argument("--no-preprocess", action="store_true",
                   help="skip all image cleanup (dewarp/deskew/contrast/threshold)")
    o.add_argument("--no-dewarp", action="store_true",
                   help="skip only perspective correction for angled photos "
                        "(keep deskew/contrast/threshold)")
    o.add_argument("--yes", "-y", action="store_true",
                   help="don't prompt for anything (writing-system codes AND the "
                        "cost confirmation); use saved values/flags and proceed")
    o.set_defaults(func=cmd_ocr)

    # build
    b = sub.add_parser("build", help="compile corrected .igt.json into a .flextext")
    b.add_argument("inputs", nargs="+", help=".igt.json files or folders")
    b.add_argument("--out", required=True, metavar="FILE", help="output .flextext path")
    b.add_argument("--font", help="font for the <languages> block")
    b.set_defaults(func=cmd_build)

    # config
    c = sub.add_parser("config", help="show or edit saved defaults")
    c.add_argument("--set", action="append", metavar="key=value",
                   help="set a config value (repeatable)")
    c.set_defaults(func=cmd_config)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
