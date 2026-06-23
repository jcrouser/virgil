from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from corpus_analyzer import CorpusAnalyzer
from enumerated_parser import EnumeratedParser

console = Console()

ENUMERATED_FAMILIES = {
    "enumerated",
    "hybrid",
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

        eligible = []

        for analysis in analyses:
            if analysis["parser_family"] in ENUMERATED_FAMILIES:
                eligible.append(
                    self.corpus_dir / analysis["filename"]
                )

        parser_engine = EnumeratedParser(self.output_dir)

        total = 0

        for pdf in eligible:
            records = parser_engine.parse_pdf(pdf)

            total += len(records)

            console.print(
                f"Parsed {pdf.name}: {len(records)} records"
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