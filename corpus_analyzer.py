from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import fitz
import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm


console = Console()


SECTION_RE = re.compile(r"^[A-Z][A-Z\s:\-&,']+$")
ENTRY_RE = re.compile(r"^\s*(\d+)\.\s+")

# A volume is "null" only when it explicitly declares that it carries no
# bibliography, e.g. Vergilius 67: "volume 67 does not contain a Vergilian
# bibliography."  An earlier version also matched the bare phrase "special
# issue", which occurs inside ordinary bibliography entries and prose (a
# journal's special issue), and wrongly discarded volumes 27, 48, and 64.
NULL_RE = re.compile(
    r"does\s+not\s+contain\s+(?:\w+\s+){0,3}bibliograph"
    r"|there\s+is\s+no\s+(?:\w+\s+){0,2}bibliograph"
    r"|no\s+(?:\w+\s+){0,2}bibliography\s+(?:was|is|will|appears)\b",
    re.IGNORECASE,
)

# Page furniture that the all-caps SECTION_RE would otherwise accept as a real
# section header. "VERGILIUS" is the journal's running head, not a category.
RUNNING_HEADER_RE = re.compile(
    r"^(?:"
    r"VERGILIUS(?:\s+\d{1,3})?"
    r"|\d{1,3}\s+VERGILIUS"
    r"|SHIRLEY\s+WERNER"
    r"|\d{1,3}\s*-\s*SHIRLEY\s+WERNER"
    r"|SHIRLEY\s+WERNER\s*-\s*\d{1,3}"
    r")$",
    re.IGNORECASE,
)


JSTOR_PATTERNS = [
    r"This content downloaded from",
    r"All use subject to",
    r"JSTOR is a not-for-profit service",
    r"Stable URL:",
]


@dataclass
class PageStats:
    page_number: int
    char_count: int
    line_count: int
    numbered_entry_lines: int
    section_headers: int
    probable_ocr_noise: float


@dataclass
class DocumentAnalysis:
    filename: str
    pages: int
    parser_family: str
    avg_chars_per_page: float
    avg_lines_per_page: float
    numbered_line_ratio: float
    section_header_count: int
    detected_sections: List[str]
    has_bibliography: bool
    probable_ocr_quality: str
    notes: List[str]


class CorpusAnalyzer:
    def __init__(self, corpus_dir: Path, output_dir: Path):
        self.corpus_dir = corpus_dir
        self.output_dir = output_dir

        self.text_dir = output_dir / "extracted_text"
        self.diag_dir = output_dir / "diagnostics"
        self.summary_dir = output_dir / "summaries"

        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    def analyze_corpus(self):
        pdfs = sorted(self.corpus_dir.glob("*.pdf"))

        analyses = []

        for pdf_path in tqdm(pdfs, desc="Analyzing PDFs"):
            try:
                analysis = self.analyze_document(pdf_path)
                analyses.append(analysis)
            except Exception as e:
                console.print(f"[red]Failed:[/red] {pdf_path.name}: {e}")

        self.write_summary(analyses)
        self.print_summary_table(analyses)

    def analyze_document(self, pdf_path: Path) -> DocumentAnalysis:
        doc = fitz.open(pdf_path)

        page_stats = []
        all_lines = []
        all_text = []
        detected_sections = set()

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            text = page.get_text("text")

            cleaned_text = self.clean_text(text)

            output_text_path = (
                self.text_dir
                / f"{pdf_path.stem}_page_{page_number + 1:03d}.txt"
            )
            output_text_path.write_text(cleaned_text, encoding="utf-8")

            lines = [line.strip() for line in cleaned_text.splitlines()]
            all_lines.extend(lines)
            all_text.append(cleaned_text)

            numbered_lines = 0
            section_headers = 0

            for line in lines:
                if ENTRY_RE.match(line):
                    numbered_lines += 1

                if self.is_section_header(line):
                    section_headers += 1
                    detected_sections.add(line)

            noise_score = self.estimate_ocr_noise(cleaned_text)

            page_stats.append(
                PageStats(
                    page_number=page_number + 1,
                    char_count=len(cleaned_text),
                    line_count=len(lines),
                    numbered_entry_lines=numbered_lines,
                    section_headers=section_headers,
                    probable_ocr_noise=noise_score,
                )
            )

        full_text = "\n".join(all_text)

        parser_family = self.classify_document(all_lines, full_text)

        # classify_document() already applies the null declaration plus the
        # numbered-entry guard, so trust its verdict rather than re-running the
        # bare regex, which would disagree with it on volumes 27, 48, and 64.
        has_bibliography = parser_family != "null"

        avg_chars = statistics.mean(p.char_count for p in page_stats)
        avg_lines = statistics.mean(p.line_count for p in page_stats)

        total_lines = sum(p.line_count for p in page_stats)
        total_numbered = sum(p.numbered_entry_lines for p in page_stats)

        numbered_ratio = total_numbered / max(total_lines, 1)

        avg_noise = statistics.mean(p.probable_ocr_noise for p in page_stats)

        ocr_quality = self.classify_ocr_quality(avg_noise)

        notes = []

        if avg_noise > 0.15:
            notes.append("High OCR noise")

        if parser_family == "hybrid":
            notes.append("Mixed structural signals")

        if not has_bibliography:
            notes.append("No bibliography detected")

        diag_path = self.diag_dir / f"{pdf_path.stem}.json"

        diag_payload = {
            "document": pdf_path.name,
            "parser_family": parser_family,
            "page_stats": [asdict(p) for p in page_stats],
            "detected_sections": sorted(detected_sections),
        }

        diag_path.write_text(
            json.dumps(diag_payload, indent=2),
            encoding="utf-8",
        )

        return DocumentAnalysis(
            filename=pdf_path.name,
            pages=len(doc),
            parser_family=parser_family,
            avg_chars_per_page=round(avg_chars, 2),
            avg_lines_per_page=round(avg_lines, 2),
            numbered_line_ratio=round(numbered_ratio, 4),
            section_header_count=len(detected_sections),
            detected_sections=sorted(detected_sections),
            has_bibliography=has_bibliography,
            probable_ocr_quality=ocr_quality,
            notes=notes,
        )

    def clean_text(self, text: str) -> str:
        lines = text.splitlines()

        cleaned = []

        for line in lines:
            skip = False

            for pattern in JSTOR_PATTERNS:
                if re.search(pattern, line):
                    skip = True
                    break

            if skip:
                continue

            line = re.sub(r"\s+", " ", line)
            line = line.strip()

            if not line:
                continue

            cleaned.append(line)

        text = "\n".join(cleaned)

        # dehyphenation
        text = re.sub(r"(\w)-\s+?(\w)", r"\1\2", text)

        return text

    def is_section_header(self, line: str) -> bool:
        if len(line) < 5:
            return False

        if len(line.split()) > 12:
            return False

        if RUNNING_HEADER_RE.match(line.strip()):
            return False

        if SECTION_RE.match(line):
            return True

        return False

    def estimate_ocr_noise(self, text: str) -> float:
        weird_chars = len(re.findall(r"[�§¤¢¥ƒ¿]", text))

        malformed_words = len(
            re.findall(r"\b\w{1,2}\*\s\w+|[A-Za-z]{1,2}\d{2,}\b", text)
        )

        total_chars = max(len(text), 1)

        noise_score = (weird_chars + malformed_words) / total_chars

        return noise_score

    def classify_ocr_quality(self, score: float) -> str:
        if score < 0.002:
            return "excellent"
        elif score < 0.01:
            return "good"
        elif score < 0.03:
            return "fair"
        else:
            return "poor"

    def classify_document(self, lines: List[str], full_text: str) -> str:
        numbered_lines = sum(1 for line in lines if ENTRY_RE.match(line))

        section_lines = sum(
            1 for line in lines if self.is_section_header(line)
        )

        prose_lines = sum(
            1
            for line in lines
            if len(line.split()) > 12 and not ENTRY_RE.match(line)
        )

        total_lines = max(len(lines), 1)

        numbered_ratio = numbered_lines / total_lines
        prose_ratio = prose_lines / total_lines

        # A null declaration only wins when the document also lacks the bulk of
        # numbered entries. This keeps a single unlucky sentence from discarding
        # a volume that plainly carries a bibliography.
        if NULL_RE.search(full_text) and numbered_lines < 20:
            return "null"

        if numbered_ratio > 0.05 and section_lines > 5:
            return "enumerated"

        if numbered_ratio > 0.01 and prose_ratio > 0.25:
            return "hybrid"

        return "narrative"

    def write_summary(self, analyses: List[DocumentAnalysis]):
        summary_path = self.summary_dir / "corpus_summary.json"

        payload = [asdict(a) for a in analyses]

        summary_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        df = pd.DataFrame(payload)

        csv_path = self.summary_dir / "corpus_summary.csv"
        df.to_csv(csv_path, index=False)

    def print_summary_table(self, analyses: List[DocumentAnalysis]):
        table = Table(title="Vergilius Corpus Analysis")

        table.add_column("Document")
        table.add_column("Parser")
        table.add_column("Pages")
        table.add_column("OCR")
        table.add_column("Sections")
        table.add_column("Notes")

        for analysis in analyses:
            table.add_row(
                analysis.filename,
                analysis.parser_family,
                str(analysis.pages),
                analysis.probable_ocr_quality,
                str(analysis.section_header_count),
                "; ".join(analysis.notes),
            )

        console.print(table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "corpus_dir",
        type=Path,
        help="Directory containing PDFs",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for analysis outputs",
    )

    args = parser.parse_args()

    analyzer = CorpusAnalyzer(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
    )

    analyzer.analyze_corpus()