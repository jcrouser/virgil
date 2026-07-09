from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional

import fitz

from enumerated_parser import (
    CurrentEntry,
    EnumeratedParser,
    _parse_entry_line,
    normalize_extracted_text,
)


@dataclass
class LayoutLine:
    page_number: int
    block_index: int
    line_index: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


class HangingIndentParser(EnumeratedParser):
    """Parser for unnumbered bibliographies separated by hanging indentation.

    Later Vergilius bibliographies do not reliably enumerate entries. Their
    record boundary signal is visual: the first line begins near the body margin,
    and continuation lines are indented. PyMuPDF's layout dictionary preserves
    that signal, so this parser only replaces record detection while reusing the
    existing field segmentation and output model from EnumeratedParser.
    """

    parser_family = "hanging_indent_layout"
    left_tolerance = 7.5
    continuation_indent = 10.0

    def _page_layout_lines(self, page, page_number: int) -> list[LayoutLine]:
        page_dict = page.get_text("dict")
        lines: list[LayoutLine] = []

        for block_index, block in enumerate(page_dict.get("blocks", [])):
            for line_index, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])
                text = normalize_extracted_text(
                    "".join(span.get("text", "") for span in spans)
                ).strip()
                if not text or self.is_noise_line(text):
                    continue

                x0, y0, x1, y1 = line.get("bbox", (0, 0, 0, 0))
                lines.append(
                    LayoutLine(
                        page_number=page_number,
                        block_index=block_index,
                        line_index=line_index,
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                        text=text,
                    )
                )

        return lines

    def _body_left_margin(self, lines: list[LayoutLine]) -> Optional[float]:
        candidates = [
            line.x0
            for line in lines
            if not self.is_section_header(line.text)
            and not self._looks_like_running_header(line.text)
            and len(line.text.split()) >= 3
        ]
        if not candidates:
            return None

        rounded = [round(x / 2) * 2 for x in candidates]
        low_band = sorted(rounded)[: max(3, len(rounded) // 3)]
        return float(median(low_band))

    def _looks_like_running_header(self, text: str) -> bool:
        text = text.strip()
        # Werner's running heads take both orders, e.g. "132 - Shirley Werner"
        # and "Vergilian Bibliography - 185". normalize_extracted_text() has
        # already folded en/em dashes to "-".
        if re.fullmatch(r"\d{1,3}\s*-\s*Shirley\s+Werner", text, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"Shirley\s+Werner\s*-\s*\d{1,3}", text, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"Vergilian\s+Bibliography\s*-\s*\d{1,3}", text, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"\d{1,3}\s*-\s*Vergilian\s+Bibliography", text, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"Vergilius(?:\s+\d{1,3})?", text, flags=re.IGNORECASE):
            return True
        if re.match(r"^\d+\s+Vergilius\b", text):
            return True
        if re.match(r"^Vergilian Bibliography\b", text, flags=re.IGNORECASE):
            return True
        if re.match(r"^Shirley\s+Werner\b", text):
            return True
        return False

    # Werner entries are author-date and open in one of a few fixed shapes.
    # Matching those shapes directly is far more reliable than looking for a
    # year on the first line: the year frequently wraps onto the indented
    # continuation line, as in
    #     Carvounis, Katerina, ... and Giampiero Scafoglio, eds.
    #         2023. Later Greek Epic and the Latin Literary Tradition. ...
    # The old rule required a year, a leading quote, or a role marker followed
    # by whitespace, so entries like the above were never recognised and their
    # text was appended to the preceding record instead.
    _SURNAME_COMMA_RE = re.compile(
        r"^\w[\w'’\-]*(?:\s+\w[\w'’\-]*){0,2},\s+\S",
        flags=re.UNICODE,
    )
    _SURNAME_YEAR_RE = re.compile(
        r"^\w[\w'’\-]*(?:\s+\w[\w'’\-]*){0,2}\.\s+(?:19|20)\d{2}\b",
        flags=re.UNICODE,
    )
    _ROLE_MARKER_RE = re.compile(r"\b(?:eds?|trans)\.(?:\s|$)", flags=re.IGNORECASE)

    def _looks_like_bibliographic_start(self, text: str) -> bool:
        text = text.strip()
        if _parse_entry_line(text) is not None:
            return False
        if self.is_section_header(text):
            return False
        if self._looks_like_running_header(text):
            return False
        # Group commentary sits at the body margin too, so it must be rejected
        # here or every commentary sentence would open a new record.
        if self._is_commentary_prose_line(text):
            return False
        if len(text.split()) < 3:
            return False

        if text.startswith('"'):
            return True
        if not (text[0].isalpha() and text[0].isupper()):
            return False

        return bool(
            self._SURNAME_COMMA_RE.match(text)
            or self._SURNAME_YEAR_RE.match(text)
            or self._ROLE_MARKER_RE.search(text)
        )

    def _is_entry_start(self, line: LayoutLine, body_left: float) -> bool:
        at_left_margin = line.x0 <= body_left + self.left_tolerance
        if not at_left_margin:
            return False
        return self._looks_like_bibliographic_start(line.text)

    def _is_commentary_line(self, line: LayoutLine, body_left: float) -> bool:
        if self._is_entry_start(line, body_left):
            return False
        if line.x0 <= body_left + self.left_tolerance and len(line.text.split()) > 16:
            return True
        return self.is_commentary(line.text)

    def _finalize_current(
        self,
        records,
        current_entry: CurrentEntry,
        page_end: int,
        source_pdf: str,
    ) -> None:
        record = self.finalize_entry(current_entry, page_end, source_pdf)
        record.parser_family = self.parser_family
        records.append(record)

    def parse_pdf(self, pdf_path: Path):
        doc = fitz.open(pdf_path)
        current_section = None
        current_entry: Optional[CurrentEntry] = None
        records = []
        synthetic_entry_number = 0

        # Group commentary state (same logic as EnumeratedParser)
        commentary_lines: list[str] = []
        commentary_page_start: int = 1
        last_entry_number: int = 0

        def flush_commentary(page_end: int) -> None:
            nonlocal commentary_lines
            if not commentary_lines:
                return
            records.append(self._make_commentary_record(
                lines=commentary_lines,
                page_start=commentary_page_start,
                page_end=page_end,
                section=current_section,
                preceding_entry_number=last_entry_number,
                source_pdf=pdf_path.name,
            ))
            commentary_lines = []

        for page_index in range(len(doc)):
            page_number = page_index + 1
            page = doc.load_page(page_index)
            layout_lines = self._page_layout_lines(page, page_number)
            body_left = self._body_left_margin(layout_lines)
            if body_left is None:
                continue

            layout_lines.sort(key=lambda line: (line.block_index, line.line_index, line.y0, line.x0))

            for line in layout_lines:
                if self.is_section_header(line.text):
                    flush_commentary(page_number)
                    current_section = line.text
                    continue

                numbered = _parse_entry_line(line.text)
                if numbered is not None:
                    flush_commentary(page_number)
                    entry_number, entry_text = numbered
                    if current_entry:
                        self._finalize_current(records, current_entry, page_number, pdf_path.name)
                        last_entry_number = current_entry.entry_number
                        current_entry = None
                    current_entry = CurrentEntry(
                        entry_number=entry_number,
                        section=current_section,
                        page_start=page_number,
                        lines=[entry_text],
                        commentary_lines=[],
                    )
                    synthetic_entry_number = max(synthetic_entry_number, entry_number)
                    continue

                if self._is_entry_start(line, body_left):
                    flush_commentary(page_number)
                    if current_entry:
                        self._finalize_current(records, current_entry, page_number, pdf_path.name)
                        last_entry_number = current_entry.entry_number
                        current_entry = None
                    synthetic_entry_number += 1
                    current_entry = CurrentEntry(
                        entry_number=synthetic_entry_number,
                        section=current_section,
                        page_start=page_number,
                        lines=[line.text],
                        commentary_lines=[],
                    )
                    continue

                if current_entry is None and not commentary_lines:
                    continue

                if not commentary_lines and self._is_commentary_prose_line(line.text):
                    if current_entry:
                        self._finalize_current(records, current_entry, page_number, pdf_path.name)
                        last_entry_number = current_entry.entry_number
                        current_entry = None
                    commentary_page_start = page_number
                    commentary_lines.append(line.text)
                    continue

                if commentary_lines:
                    commentary_lines.append(line.text)
                    continue

                if self._is_commentary_line(line, body_left):
                    current_entry.commentary_lines.append(line.text)
                else:
                    current_entry.lines.append(line.text)

        flush_commentary(len(doc))
        if current_entry:
            self._finalize_current(records, current_entry, len(doc), pdf_path.name)

        output_path = self.record_dir / f"{pdf_path.stem}.json"
        output_path.write_text(
            json.dumps([record.model_dump() for record in records], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return records


if __name__ == "__main__":
    import argparse

    from enumerated_parser import write_records_csv

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--csv-name",
        default="hanging_indent_records.csv",
        help="Name of the combined CSV written to output_dir.",
    )

    args = parser.parse_args()

    engine = HangingIndentParser(args.output_dir)
    all_records = []

    for pdf_path in sorted(args.pdf_dir.glob("*.pdf")):
        records = engine.parse_pdf(pdf_path)
        all_records.extend(records)

    write_records_csv(all_records, args.output_dir / args.csv_name)
