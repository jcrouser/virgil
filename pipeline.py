from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from corpus_analyzer import CorpusAnalyzer
from enumerated_parser import EnumeratedParser, write_records_csv
from hanging_indent_parser import HangingIndentParser

console = Console()

ENUMERATED_FAMILIES = {
    "enumerated",
    "hybrid",
}

LAYOUT_FAMILIES = {
    "narrative",
}


class VergiliusPipeline:
    def __init__(self, corpus_dir: Path, output_dir: Path):
        self.corpus_dir = corpus_dir
        self.output_dir = output_dir

    def run(self):
        analyzer = CorpusAnalyzer(
            self.corpus_dir,
            self.output_dir,
        )

        analyzer.analyze_corpus()

        summary_path = (
            self.output_dir
            / "summaries"
            / "corpus_summary.json"
        )

        analyses = json.loads(summary_path.read_text())

        parse_jobs = []
        skipped = []

        for analysis in analyses:
            if analysis["parser_family"] in ENUMERATED_FAMILIES:
                parse_jobs.append(
                    (
                        self.corpus_dir / analysis["filename"],
                        "enumerated",
                    )
                )
            elif (
                analysis["parser_family"] in LAYOUT_FAMILIES
                and analysis["has_bibliography"]
            ):
                parse_jobs.append(
                    (
                        self.corpus_dir / analysis["filename"],
                        "hanging_indent",
                    )
                )
            else:
                skipped.append(
                    {
                        "filename": analysis["filename"],
                        "reason": analysis["parser_family"],
                    }
                )

        parser_engines = {
            "enumerated": EnumeratedParser(self.output_dir),
            "hanging_indent": HangingIndentParser(self.output_dir),
        }

        total = 0
        all_records = []
        parsed_documents = []

        for pdf, parser_name in parse_jobs:
            records = parser_engines[parser_name].parse_pdf(pdf)

            total += len(records)
            all_records.extend(records)
            parsed_documents.append(
                {
                    "filename": pdf.name,
                    "parser": parser_name,
                    "records": len(records),
                    "status": "parsed",
                }
            )

            console.print(
                f"Parsed {pdf.name} with {parser_name}: {len(records)} records"
            )

        # Corpus-specific CSV, e.g. "McKay" → output/mckay_records.csv
        corpus_name = self.corpus_dir.name.lower()
        write_records_csv(
            all_records,
            self.output_dir / f"{corpus_name}_records.csv",
        )

        pipeline_summary = {
            "total_records": total,
            "parsed_documents": parsed_documents,
            "skipped_documents": skipped,
        }

        summary_output_path = (
            self.output_dir
            / "summaries"
            / "pipeline_summary.json"
        )
        summary_output_path.write_text(
            json.dumps(pipeline_summary, indent=2),
            encoding="utf-8",
        )

        console.print(f"\nTotal extracted records: {total}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("output_dir", type=Path)

    args = parser.parse_args()

    pipeline = VergiliusPipeline(
        args.corpus_dir,
        args.output_dir,
    )

    pipeline.run()
