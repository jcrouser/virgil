"""Combine per-PDF JSON records from multiple pipeline runs into one CSV.

Each ./run.sh call overwrites corpus_summary.json and pipeline_summary.json
with only that batch's data, but per-PDF JSON files in parsed_records/ and
per-PDF diagnostic files in diagnostics/ accumulate across runs.

This script reads everything that accumulated and writes:
  output/records.csv            — all bibliographic + commentary records
  output/summaries/corpus_summary.json   — rebuilt from per-PDF diagnostics
  output/summaries/corpus_summary.csv    — same, in CSV form
  output/summaries/pipeline_summary.json — rebuilt from per-PDF JSON records

Usage:
    python combine.py              # uses output/
    python combine.py my_output    # custom output directory
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from enumerated_parser import BibliographyRecord, write_records_csv

console = Console()


def _rebuild_corpus_summary(output_dir: Path) -> list[dict]:
    """Rebuild corpus_summary from the per-PDF diagnostic files."""
    diag_dir = output_dir / "diagnostics"
    if not diag_dir.exists():
        return []

    summaries = []
    for df in sorted(diag_dir.glob("*.json")):
        try:
            diag = json.loads(df.read_text(encoding="utf-8"))
        except Exception:
            continue

        summaries.append({
            "filename": diag.get("document", df.stem + ".pdf"),
            "parser_family": diag.get("parser_family", "unknown"),
            "section_header_count": len(diag.get("detected_sections", [])),
            "detected_sections": diag.get("detected_sections", []),
            "page_stats": diag.get("page_stats", []),
        })

    return summaries


def combine(output_dir: Path) -> None:
    records_dir = output_dir / "parsed_records"
    summaries_dir = output_dir / "summaries"

    if not records_dir.exists():
        console.print(f"[red]No parsed_records folder found at {records_dir}[/red]")
        raise SystemExit(1)

    json_files = sorted(records_dir.glob("*.json"))
    if not json_files:
        console.print(f"[red]No JSON files found in {records_dir}[/red]")
        raise SystemExit(1)

    # ------------------------------------------------------------------ #
    # 1. Combine all records                                               #
    # ------------------------------------------------------------------ #
    all_records: list[BibliographyRecord] = []
    type_counts: Counter[str] = Counter()
    parsed_documents: list[dict] = []

    for jf in json_files:
        try:
            raw = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            console.print(f"[yellow]Skipping {jf.name}: {e}[/yellow]")
            continue

        file_records = []
        for item in raw:
            all_records.append(BibliographyRecord(**item))
            type_counts[item.get("entry_type", "unknown")] += 1
            file_records.append(item)

        if file_records:
            parser_used = file_records[0].get("parser_family", "unknown")
            parsed_documents.append({
                "filename": jf.stem + ".pdf",
                "parser": parser_used,
                "records": len(file_records),
                "status": "parsed",
            })

    csv_path = output_dir / "records.csv"
    write_records_csv(all_records, csv_path)

    # ------------------------------------------------------------------ #
    # 2. Write combined pipeline_summary.json                             #
    # ------------------------------------------------------------------ #
    summaries_dir.mkdir(parents=True, exist_ok=True)

    pipeline_summary = {
        "total_records": len(all_records),
        "parsed_documents": parsed_documents,
    }
    (summaries_dir / "pipeline_summary.json").write_text(
        json.dumps(pipeline_summary, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # 3. Rebuild corpus_summary from per-PDF diagnostics                  #
    # ------------------------------------------------------------------ #
    corpus_summary = _rebuild_corpus_summary(output_dir)
    if corpus_summary:
        (summaries_dir / "corpus_summary.json").write_text(
            json.dumps(corpus_summary, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(corpus_summary).to_csv(
            summaries_dir / "corpus_summary.csv", index=False
        )

    # ------------------------------------------------------------------ #
    # 4. Print report                                                      #
    # ------------------------------------------------------------------ #
    table = Table(title="Combined Output", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Records", justify="right")

    for doc in parsed_documents:
        table.add_row(doc["filename"], str(doc["records"]))

    console.print(table)
    console.print()
    console.print(f"PDFs combined:  {len(json_files)}")
    console.print(f"Total records:  {len(all_records)}")
    console.print(f"CSV written to: {csv_path}")
    console.print()

    for entry_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        console.print(f"  {entry_type:<30} {count:>6}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine per-PDF JSON records from multiple pipeline runs."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="output",
        type=Path,
        help="Output directory used by the pipeline runs (default: output)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir

    combine(output_dir)
