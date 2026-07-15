# ocr-igt

OCR of sloppy, handwritten **Fayu** field-notebook scans into a FLEx-importable
`.flextext` interlinear file. Fayu baseline, Indonesian glosses (often misspelled
and run together), Indonesian free translations.

The workflow is deliberately **two steps**, because the scans are messy and OCR
is imperfect:

```
ocr-igt ocr   <images/PDFs>   →   editable  *.igt.json   (correct these by hand)
ocr-igt build <corrected json> →  one  .flextext         (import into FLEx)
```

Two pluggable engines:

| Engine       | `--engine`   | Runs where | Notes |
|--------------|--------------|------------|-------|
| **Tesseract** (default) | `tesseract` | fully offline, no cost | Weak on sloppy handwriting — gives a rough scaffold to fix by hand. Keeps the raw OCR lines and flags every guess. |
| **Claude vision** | `claude` | Anthropic API | Much better on low-literacy handwriting: it's told the interlinear structure and returns aligned word↔gloss JSON. Sends images to the API; ~$0.05–0.10/page. |

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

`pip install -e .` puts the **`ocr-igt`** command on your PATH, so you can run
`ocr-igt …` from anywhere. Prefer not to install? Run it as a module from the
project directory instead: `python -m ocr_igt …` (equivalent to every `ocr-igt`
command below).

### Tesseract engine (offline) — extra system dependency

`pytesseract` needs the **`tesseract` binary** plus Indonesian training data:

```bash
# macOS
brew install tesseract tesseract-lang     # tesseract-lang includes 'ind'

# Debian/Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-ind
```

If `ind` isn't installed the app falls back to `eng`.

### Claude engine — API key & cost

Set an API key (never stored by this tool):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

(An `ant auth login` profile also works — the SDK picks it up automatically.)

**Cost.** Each page is one vision request on `claude-opus-4-8`
(\$5 / \$25 per 1M input/output tokens). A preprocessed page (~1568 px long
edge) plus the prompt is ≈ 3–4k input tokens, and a page of IGT is ≈ 1–2k output
tokens — roughly **\$0.05–0.10 per page**, so a 100-page notebook is on the order
of **\$5–10**. The tool prints real token usage and an estimated cost after every
page and a running total at the end, so you're never guessing. Use
`--model claude-sonnet-5` to cut cost roughly in half at some accuracy loss.

> Your scans are sent to the Anthropic API only when you choose `--engine claude`.
> The Tesseract engine keeps everything on your machine.

---

## Usage

### 1. OCR → editable JSON

```bash
# Offline Tesseract (the default engine), a whole folder of scans:
ocr-igt ocr ./notebook-scans --out ./work

# Claude vision (recommended for handwriting), a PDF:
ocr-igt ocr notebook.pdf --engine claude --out ./work

# A folder, but skip a file (e.g. a cover / work-hours page):
ocr-igt ocr ./notebook-scans --engine claude --exclude tosokai_0.jpg --out ./work
```

**Choosing the engine.** `--engine claude` uses the vision API (best for
handwriting); `--engine tesseract` runs fully offline (default). The choice is
**saved**, so it sticks until you pass `--engine` again — or set it permanently
with `ocr-igt config --set engine=claude`.

**Excluding files.** `--exclude` is repeatable and matches by exact filename,
by stem, or as a glob — `--exclude tosokai_0.jpg`, `--exclude tosokai_0`, and
`--exclude 'tosokai_*cover*'` all work. Or just list the files you *do* want
explicitly instead of pointing at the folder.

**Cost estimate + confirmation (Claude engine).** Before sending anything, it
prints an estimated cost, the amount `ocr-igt` has spent so far (a local tally —
see below), and waits for a `y/N` confirmation:

```
8 page(s) will be sent to the Anthropic API (claude-opus-4-8).
  Estimated cost:  ~$0.34   (~12,427 input + ~11,200 output tokens; rough, usually high)
  Spent by ocr-igt so far (local tally): $0.28
  Live account balance isn't available via the API — check https://console.anthropic.com/settings/billing
Proceed? [y/N]:
```

Pass `--yes` (or `-y`) to skip the confirmation (and the writing-system prompt)
for scripting. In a non-interactive shell the run **aborts** unless `--yes` is
given, so a background job can never silently spend.

> **On "remaining balance":** Anthropic doesn't expose your prepaid balance
> through the API (there's no balance endpoint — it 404s), so the tool can't show
> it. What it *can* do is tally its own spend in `~/.config/ocr-igt/spend.log`
> and show a running total. For your real balance, use the
> [Console billing page](https://console.anthropic.com/settings/billing).

On the first run it **asks for the writing-system codes** your FLEx project uses
and remembers them for next time (press Enter to accept the saved default):

```
Writing-system codes for this FLEx project (press Enter to accept the saved default):
  Vernacular (Fayu baseline) [fau]:
  Gloss + free-translation (Indonesian) [id]:
  Analysis (segment numbers) [en]:
```

Pass them as flags to skip the prompt: `--vern fau --gloss id --analysis en`, or
add `--yes` to accept saved values non-interactively.

### 2. Correct the JSON

Open the `*.igt.json` sidecars and fix the OCR. Each phrase is one glossed line:

```json
{
  "source_image": "page012.png",
  "engine": "claude",
  "vernacular": "fau",
  "gloss_lang": "id",
  "analysis_lang": "en",
  "title": "12",
  "phrases": [
    {
      "words": [
        { "txt": "abogo", "gloss": "pergi" },
        { "txt": "sai",   "gloss": "dia" }
      ],
      "free": "Dia pergi.",
      "note": ""
    }
  ]
}
```

- `words[].txt` = Fayu word, `words[].gloss` = Indonesian gloss (word-level).
- `free` = Indonesian free translation.
- `note`, `raw_lines`, `warnings` are for your reference. `note` (if kept) becomes
  a phrase-level FLEx note; `raw_lines`/`warnings` are not exported.

### 3. Build the `.flextext`

```bash
ocr-igt build ./work --out fayu-notebook.flextext
```

Then in FLEx: **File ▸ Import ▸ FLExText Interlinear**. The importer uses the
writing-system codes from the JSON, so make sure they match your project's
writing systems (adjust with `--vern/--gloss/--analysis` at OCR time, or edit the
JSON, or `ocr-igt config`).

### Config

```bash
ocr-igt config                       # show saved defaults + file location
ocr-igt config --set font="Doulos SIL" --set model=claude-sonnet-5
```

---

## How the output maps to FLEx

- Each page → one `<interlinear-text>`.
- Each phrase → a `<phrase>` with a `segnum`, its `words`, and the free
  translation as the phrase-level `<item type="gls">`.
- Each word → `<item type="txt">` (Fayu) + `<item type="gls">` (Indonesian
  word gloss). Trailing punctuation is split into `<item type="punct">` words.

This imports as a baseline + word-gloss + free-translation interlinear. Morpheme
segmentation is left to FLEx — glossing here is at the word level, which is what
a handwritten gloss line gives you.

## Limitations & tips

- **Tesseract on handwriting is genuinely poor.** Expect to rewrite most of it.
  Use it for offline drafts; use `--engine claude` when accuracy matters.
- **Glosses that run together** are the hard case. The Claude engine is asked to
  segment and align them and to flag low-confidence alignments in `note`; the
  Tesseract engine aligns by whitespace and flags count mismatches. Always
  eyeball the alignment.
- **Nothing is auto-corrected.** Misspelled Indonesian and non-standard Fayu are
  transcribed as written — fixing them is a linguistic decision, left to you.
- Re-run `build` freely; it just recompiles whatever JSON you point it at.
