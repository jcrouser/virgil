# Virgil — Vergilian Bibliography Parser

Extracts structured bibliographic records from the annual "Vergilian Bibliography" sections of the *Vergilius* journal (volumes 10–71, 1964–2025), compiled first by Alexander McKay and then by Shirley Werner. Output is per-PDF JSON and per-corpus CSV.

---

## Quick start

```bash
# Parse each corpus folder into output/
./run.sh McKay output        # → output/mckay_records.csv
./run.sh Enumerated output   # → output/enumerated_records.csv
./run.sh Werner output       # → output/werner_records.csv

# Combine all three into one spreadsheet
python combine.py            # → output/records.csv
```

`run.sh` creates and maintains a `.venv` automatically on first run.

---

## Corpus layout

```
corpus/
  McKay/        volumes 10–44  (1964–1998)  enumerated format
  Enumerated/   volumes 45–55  (1999–2009)  enumerated format (McKay → Werner transition)
  Werner/       volumes 56–71  (2010–2025)  hanging-indent format
```

PDFs follow the naming convention `NN (YYYY) - Author.pdf`.

---

## Output layout

Each `./run.sh` call writes into the output directory:

```
output/
  <corpus>_records.csv          per-corpus spreadsheet (mckay / enumerated / werner)
  parsed_records/               one JSON file per PDF
  diagnostics/                  per-PDF quality metrics
  extracted_text/               per-page cleaned text
  summaries/
    corpus_summary.json/csv     document classification for that run
    pipeline_summary.json       record counts for that run

# After running combine.py:
  records.csv                   all corpora combined
```

---

## Source files

| File | Role |
|---|---|
| `run.sh` | Entry point. Manages `.venv`, resolves corpus path shortcuts, calls `pipeline.py`. |
| `pipeline.py` | Orchestrator. Runs `CorpusAnalyzer`, routes each PDF to the right parser, writes the per-corpus CSV and `pipeline_summary.json`. |
| `corpus_analyzer.py` | Stage 1. Extracts text from every PDF, classifies it as `enumerated`, `hybrid`, `narrative`, or `null`, and writes `corpus_summary.json` and per-page text files. |
| `enumerated_parser.py` | Core parser for **numbered** bibliography entries (McKay vols 10–55). Handles PDF text-layer repairs (mojibake, shifted ASCII, Cyrillic lookalikes, line-break hyphenation), detects entry boundaries, segments each entry into author / year / title / publication / reviews, and emits `group_commentary` records for prose paragraphs that annotate multiple entries at once. |
| `hanging_indent_parser.py` | Parser for **unnumbered** entries (Werner vols 56–71). Uses PyMuPDF bounding boxes to detect entry starts by left-margin position instead of entry numbers. Inherits all segmentation logic from `enumerated_parser.py`. |
| `combine.py` | Post-processing utility. Reads all accumulated per-PDF JSONs from `parsed_records/`, writes a combined `records.csv`, and rebuilds `pipeline_summary.json` and `corpus_summary.json` across all three runs. |

---

## CSV columns

| Column | Description |
|---|---|
| `source_pdf` | Source filename |
| `entry_number` | Entry number in the original PDF (synthetic for hanging-indent volumes) |
| `entry_type` | `monograph`, `journal_article`, `edited_volume`, `dissertation`, `translation`, `unknown`, or `group_commentary` |
| `authors` | Pipe-separated list of author names |
| `year` | Publication year |
| `title` | Article or book title |
| `publication_block` | Publisher, place, journal citation, series |
| `commentary` | Text of the record if `entry_type` is `group_commentary` |
| `reviews` | Pipe-separated review citations |
| `confidence` | 0–1 score based on how many fields were successfully extracted |
| `segment_warning` | Flag when a field boundary was inferred rather than explicit |
| `raw_text` | Normalised single-line version of the extracted text |
| `raw_text_original` | Preserves PDF line breaks for auditing |

`group_commentary` rows contain prose paragraphs that discuss several entries at once. Their `entry_number` matches the last numbered entry before the paragraph.

---

## Dependencies

```
PyMuPDF   — PDF text and layout extraction
pandas    — CSV output
pydantic  — record validation and serialisation
rich      — progress display
tqdm      — progress bars
```

Install automatically via `run.sh`, or manually:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
