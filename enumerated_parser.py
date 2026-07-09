"""Enumerated Werner bibliography parser.

This script parses the *numbered* Shirley Werner bibliography PDFs, especially
volumes 53--55, into per-PDF JSON files and one combined CSV. It is intentionally
not a fully general bibliography parser. Instead, it encodes the observed grammar
of the numbered Werner entries:

    <entry number>. <authors/editors>. <year>. <title>. <publication tail>. Rev: ...

The parser is organized as a pipeline:

1. Extract raw text from PDFs with PyMuPDF.
2. Repair predictable PDF/text-layer problems: mojibake, Cyrillic lookalikes,
   soft hyphens, curly quotes, JSTOR footer noise, and line-break hyphenation.
3. Detect enumerated entries and accumulate continuation lines.
4. Segment each entry into authors, year, title, publication metadata, reviews,
   and optional commentary.
5. Serialize archival JSON plus a spreadsheet-friendly CSV.

Most future tuning should happen in three places:

* _parse_entry_line() for entry-boundary errors.
* segment_entry() for high-level bibliographic grammar.
* _split_by_publication_tail() for title/publication-block boundary errors.

The code favors conservative, auditable parsing over aggressive normalization:
`raw_text_original` preserves the PDF line breaks, while `raw_text` is the
normalized single-line form used for segmentation.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import fitz
from pydantic import BaseModel
from rich.console import Console
from rich.progress import track

console = Console()


# -----------------------------------------------------------------------------
# Unicode / PDF text repair
# -----------------------------------------------------------------------------

MOJIBAKE_MARKERS = (
    "√",   # MacRoman-decoded UTF-8, e.g. √© for é
    "Ã",   # Latin-1/Windows-1252-decoded UTF-8, e.g. Ã© for é
    "Â",   # Often appears before non-breaking spaces or symbols
    "â€",  # Smart quotes / dashes decoded as Windows-1252/Latin-1
    "�",   # Unicode replacement character
)


def _mojibake_score(text: str) -> int:
    """Count visible markers that usually indicate decoded-with-the-wrong-codec text."""
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def _try_redecode(text: str, wrong_encoding: str) -> str | None:
    """Try to reverse mojibake by re-encoding through the suspected wrong codec."""
    try:
        return text.encode(wrong_encoding).decode("utf-8")
    except UnicodeError:
        return None


def repair_mojibake(text: str) -> str:
    """Repair common mojibake patterns without changing already-clean text.

    Examples include MacRoman-style `√©` and cp1252/latin-1-style `Ã©`.
    The function tries candidate repairs and keeps the one with fewer mojibake
    markers, which makes the operation safe enough to run on every line.
    """
    if not text:
        return text

    text = unicodedata.normalize("NFC", text)
    original_score = _mojibake_score(text)
    if original_score == 0:
        return text

    best = text
    best_score = original_score

    encodings: list[str] = []
    if "√" in text:
        encodings.append("mac_roman")
    if "Ã" in text or "Â" in text or "â€" in text:
        encodings.extend(["cp1252", "latin-1"])

    for wrong_encoding in dict.fromkeys(encodings):
        candidate = _try_redecode(text, wrong_encoding)
        if candidate is None:
            continue
        candidate = unicodedata.normalize("NFC", candidate)
        score = _mojibake_score(candidate)
        if score < best_score:
            best = candidate
            best_score = score

    return best


# Some PDFs in this corpus appear to substitute visually similar Cyrillic glyphs
# into otherwise Latin text, especially in initials. Repair the common cases so
# "В. G. F." becomes "B. G. F." and "R. О. A." becomes "R. O. A.".
CYRILLIC_LATIN_CONFUSABLES = str.maketrans({
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K",
    "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
})


def repair_confusable_glyphs(text: str) -> str:
    """Replace common Cyrillic lookalikes with Latin characters."""
    if not text:
        return text
    return text.translate(CYRILLIC_LATIN_CONFUSABLES)


SHIFTED_ASCII_MARKERS = (
    "\x03",  # space encoded 29 code points too low
    "\x15\x13",  # years beginning with 20..
    "9HUJ",  # Verg...
    "6KLUOH",  # Shirle...
    "ELEOLRJUDSK",  # bibliograph...
    "DQG",  # and
    "HGV",  # eds
    "WKH",  # the
    "LQ",  # in
    "RI",  # of
    "0DLD",  # Maia
    "5LY",  # Riv...
    "0L",  # Mi...
    "$",  # A
)

HIGH_SHIFTED_ASCII_MARKERS = (
    "OMOP",  # 2023
    "sergil",  # Vergil...
    "tilli",  # Willi...
    "compil~tion",  # compilation
)


def _shifted_ascii_score(text: str) -> int:
    control_chars = len(re.findall(r"[\x01-\x1f]", text))
    visible_markers = sum(text.count(marker) for marker in SHIFTED_ASCII_MARKERS)
    return control_chars + visible_markers * 8


def _low_shifted_visible_marker_score(text: str) -> int:
    return sum(
        text.count(marker)
        for marker in SHIFTED_ASCII_MARKERS
        if marker != "\x03"
    )


def _high_shifted_ascii_score(text: str) -> int:
    visible_markers = sum(
        text.count(marker)
        for marker in HIGH_SHIFTED_ASCII_MARKERS
    )
    separator_markers = text.count("=") + text.count("~")
    return visible_markers * 8 + separator_markers


def _high_shifted_visible_marker_score(text: str) -> int:
    return sum(text.count(marker) for marker in HIGH_SHIFTED_ASCII_MARKERS)


def repair_shifted_ascii_text_layer(text: str) -> str:
    """Repair PDFs whose ToUnicode map emits ASCII 29 code points too low.

    Some recent JSTOR/PDF text layers encode ordinary ASCII shifted by 29 code
    points. The corpus contains both directions: ``9HUJLOLDQ`` is ``Vergilian``
    in one map, while ``sergilius`` is ``Vergilius`` in another. These repairs
    are gated by visible markers so ordinary text is left alone.
    """
    if not text:
        return text

    low_score = _shifted_ascii_score(text)
    if low_score >= 8 and _low_shifted_visible_marker_score(text):
        repaired = []
        for index, char in enumerate(text):
            code = ord(char)
            previous_char = text[index - 1] if index else ""
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if char in "\n\r\t":
                repaired.append(char)
            elif char == " ":
                repaired.append(char)
            elif char.isdigit():
                if next_char.isalpha() or previous_char.isalpha():
                    repaired.append(chr(code + 29))
                else:
                    repaired.append(char)
            elif char == ":":
                if next_char.isupper():
                    repaired.append(chr(code + 29))
                else:
                    repaired.append(char)
            elif char in ".,;?!'\"()[]-":
                repaired.append(char)
            elif char.isupper():
                if (
                    previous_char.isupper()
                    or next_char.isupper()
                    or previous_char.isdigit()
                    or (previous_char and previous_char in "$&*/:")
                ):
                    repaired.append(chr(code + 29))
                else:
                    repaired.append(char)
            elif char.islower():
                repaired.append(char)
            elif 1 <= code <= 97:
                repaired.append(chr(code + 29))
            else:
                repaired.append(char)

        candidate = "".join(repaired)
        if _shifted_ascii_score(candidate) < low_score:
            text = candidate
    elif "\x03" in text:
        text = text.replace("\x03", " ")

    high_score = _high_shifted_ascii_score(text)
    if high_score < 8 or not _high_shifted_visible_marker_score(text):
        return text

    repaired = []
    for char in text:
        code = ord(char)
        if char in "\n\r\t":
            repaired.append(char)
        elif 61 <= code <= 126:
            repaired.append(chr(code - 29))
        else:
            repaired.append(char)

    candidate = "".join(repaired)
    if _high_shifted_ascii_score(candidate) < high_score:
        return candidate
    return text


def normalize_extracted_text(text: str) -> str:
    """Normalize the raw PyMuPDF text layer before any parsing decisions.

    This function intentionally performs only low-level text cleanup. It does
    not flatten line breaks, because line breaks are still useful when deciding
    whether a quotation mark was dropped or a word was hyphenated across lines.
    """
    text = repair_shifted_ascii_text_layer(text)
    text = repair_mojibake(text)
    text = repair_confusable_glyphs(text)
    text = text.replace("\x00", "")
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    return unicodedata.normalize("NFC", text)




# Letter class used for PDF line-break dehyphenation. This covers ordinary Latin
# letters plus most accented Latin characters found in classical bibliography.
LATIN_LETTER_CLASS = r"A-Za-zÀ-ÖØ-öø-ÿĀ-žḀ-ỿ"


def dehyphenate_pdf_linebreaks(text: str) -> str:
    """
    Repair words split by a PDF line break, e.g.:
        "Pe-\nrutelli" -> "Perutelli"

    This is intentionally conservative: it only removes a hyphen when it is
    between letters across an actual newline. Hyphens inside a line are kept.
    """
    if not text:
        return text
    text = normalize_extracted_text(text)
    pattern = rf"(?<=[{LATIN_LETTER_CLASS}])-\s*\n\s*(?=[{LATIN_LETTER_CLASS}])"
    return re.sub(pattern, "", text)


def normalize_for_parsing(text: str) -> str:
    """Normalize Unicode and repair line-break hyphenation before segmentation.

    This preserves non-hyphen line breaks. Use normalize_entry_text() when you
    want a single-line bibliographic record for output or field segmentation.
    """
    return dehyphenate_pdf_linebreaks(normalize_extracted_text(text))


def normalize_entry_text(text: str) -> str:
    """Return a single-line version of an extracted bibliography entry.

    Processing order matters:
    1. Repair mojibake / Unicode oddities.
    2. Remove hyphenation that occurs across actual PDF line breaks.
    3. Convert remaining line breaks to spaces.
    4. Collapse repeated horizontal whitespace.

    Examples:
        "Pe-\nrutelli" -> "Perutelli"
        "Vergilian Bibliography 200\nVergilius 52" ->
            "Vergilian Bibliography 200 Vergilius 52"
    """
    if not text:
        return text
    text = dehyphenate_pdf_linebreaks(normalize_extracted_text(text))
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return unicodedata.normalize("NFC", text).strip()

# -----------------------------------------------------------------------------
# Regexes and constants
# -----------------------------------------------------------------------------

# Entry numbers in these PDFs are mostly ordinary "15." strings, but OCR/text
# extraction occasionally inserts spaces: "1 5 .". A year at the start of a
# continuation line ("2006. Title...") should not become entry #2006.
# The leading \.? handles rare OCR artifacts where the previous line's terminal
# period bleeds onto the next line: ".50. Author..." should match as entry 50.
ENTRY_RE = re.compile(r"^\s*\.?\s*((?:\d\s*){1,4})\.\s+(.*)")
MAX_REASONABLE_ENTRY_NUMBER = 1000
SECTION_RE = re.compile(r"^[A-Z][A-Z\s:\-&,']+$")

# Page furniture. "VERGILIUS" is the journal's running head; SECTION_RE would
# otherwise accept it as a section and it would overwrite the real category for
# every entry on the page. Dashes are already folded to "-" by
# normalize_extracted_text(), so only the ASCII hyphen needs matching here.
RUNNING_HEADER_RE = re.compile(
    r"^(?:"
    r"VERGILIUS(?:\s+\d{1,3})?"
    r"|\d{1,3}\s+VERGILIUS"
    r"|SHIRLEY\s+WERNER"
    r"|\d{1,3}\s*-\s*SHIRLEY\s+WERNER"
    r"|SHIRLEY\s+WERNER\s*-\s*\d{1,3}"
    r"|VERGILIAN\s+BIBLIOGRAPHY\s*-\s*\d{1,3}"
    r"|\d{1,3}\s*-\s*VERGILIAN\s+BIBLIOGRAPHY"
    r")$",
    flags=re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
REVIEW_RE = re.compile(r"\bRev(?:iews?)?:\s*(.*)", flags=re.IGNORECASE)

# Common publication/opening tokens. These are not meant to be exhaustive; they
# are used only as conservative split cues after author/title detection.
CITY_TOKENS = {
    "Oxford",
    "Cambridge",
    "London",
    "Leiden",
    "Boston",
    "Paris",
    "Roma",
    "Milano",
    "New York",
    "Stuttgart",
    "Remshalden",
    "Swansea, Wales",
    "Swansea",
    "Wales",
    "Berlin",
    "Munich",
    "Amsterdam",
    "Princeton",
    "Cambridge, MA",
    "Bruxelles",
    "Brussels",
    "Napoli",
    "Bari",
    "Pisa",
    "Firenze",
    "Bologna",
    "Turnhout",
    "Göttingen",
    "Tübingen",
}

JOURNAL_OR_SERIES_CUES = (
    "AJP", "AJPh", "Arethusa", "BMCR", "CJ", "CP", "CQ", "CR", "CW", "G&R",
    "HSCP", "JRS", "Latomus", "MD", "Mnemosyne", "Phoenix", "TAPA", "Vergilius",
    "Vergil", "Aevum", "Athenaeum", "Hermes", "Ramus", "Rheinisches Museum",
    "Classical World", "Classical Journal", "Classical Philology",
)

PUB_CUE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(cue) for cue in sorted(JOURNAL_OR_SERIES_CUES, key=len, reverse=True)) + r")\b"
)

# Series / publication-block openings that may themselves contain city names
# (e.g., "Proceedings of the Cambridge Philological Society..."). These must
# be detected before blindly splitting at the first city token.
PUBLICATION_START_RE = re.compile(
    r"\.\s+(?=(?:"
    r"Proceedings\s+of\b|"
    r"Proc\.\b|"
    r"Transactions\s+of\b|"
    r"Suppl\.\b|"
    r"Supplement\b|"
    r"Supplementum\b|"
    r"Collection\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'&\- ]+\s+\d+\b|"
    r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'&\- ]+\s+Collection\s+\d+\b|"
    r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'&\- ]+\s+Suppl\.\b|"
    r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'&\- ]+\s+Supplement\b|"
    r"Vol\.\s*\d+\b|"
    r"\d+\s+vols?\.?\b"
    r"))",
    flags=re.IGNORECASE,
)

# Third-person verbs that appear in McKay/Werner group commentary sentences,
# e.g. "Geymonat favours...", "Cicu analyses...", "Bauzá is concerned with...".
# Used by EnumeratedParser._is_commentary_prose_line() to distinguish prose
# paragraphs from bibliographic entry continuation lines.
_COMMENTARY_VERBS: frozenset[str] = frozenset({
    "offers", "offer", "provides", "provide", "argues", "argue",
    "explores", "explore", "studies", "study", "favours", "favors",
    "favour", "favor", "gives", "give", "discusses", "discuss",
    "examines", "examine", "analyses", "analyzes", "analyze", "analyse",
    "suggests", "suggest", "takes", "take", "outlines", "outline",
    "considers", "consider", "notes", "note", "presents", "present",
    "treats", "treat", "defends", "defend", "proposes", "propose",
    "attacks", "attack", "accepts", "accept", "rejects", "reject",
    "surveys", "survey", "reviews", "review", "traces", "trace",
    "compares", "compare", "interprets", "interpret", "concludes",
    "conclude", "maintains", "maintain", "shows", "show", "focuses",
    "focusses", "focus", "highlights", "highlight", "deals", "deal",
    "addresses", "address", "stresses", "stress", "detects", "detect",
    "observes", "observe", "establishes", "establish", "classifies",
    "classify", "points", "point", "remarks", "remark", "finds", "find",
    "follows", "follow", "reads", "read", "dates", "date", "identifies",
    "identify", "believes", "believe", "claims", "claim", "contends",
    "contend", "supports", "support", "doubts", "doubt", "questions",
    "question", "places", "place", "attempts", "attempt", "denies", "deny",
    "continues", "continue", "seems", "seem", "appears", "appear",
    "is", "are", "was", "were", "has", "have", "had",
    "demonstrates", "demonstrate", "illustrates", "illustrate",
    "asserts", "assert", "affirms", "affirm", "insists", "insist",
    "challenges", "challenge", "contradicts", "contradict",
    "introduces", "introduce", "concentrates", "concentrate",
    "attempts", "attempt", "seeks", "seek", "tries", "try",
})

# Name-ish author block before a title. Useful for entries whose first year is
# in the journal/publication citation, not after the author.
AUTHOR_BEFORE_QUOTE_RE = re.compile(r"^\s*(?P<author>.+?)\.\s+\"")

# Entries without a visible publication year sometimes begin with an editor role
# marker before the title, e.g.:
#     Craig W. Kallendorf, ed. A Companion to the Classical Tradition
# Split these before the generic first-period fallback, which would otherwise
# mistake the initial period in "W." for an author/title boundary.
AUTHOR_ROLE_PREFIX_RE = re.compile(
    rf"^\s*(?P<author>.+?),\s*(?P<role>eds?|trans)\.\s+(?P<rest>.+)$",
    flags=re.IGNORECASE,
)


class EntryType(str, Enum):
    """Controlled vocabulary for the broad bibliographic item type."""
    MONOGRAPH = "monograph"
    JOURNAL_ARTICLE = "journal_article"
    EDITED_VOLUME = "edited_volume"
    DISSERTATION = "dissertation"
    TRANSLATION = "translation"
    UNKNOWN = "unknown"


class Review(BaseModel):
    """A review citation attached to an entry after `Rev:` / `Review:`."""
    raw_text: str


class BibliographicSegments(BaseModel):
    """Intermediate segmentation result before authors/title/year are finalized.

    Keeping this as a separate model makes it easier to debug whether an error
    happened during segmentation or during later normalization/classification.
    """
    raw_text: str
    author_block: Optional[str] = None
    year_block: Optional[str] = None
    title_block: Optional[str] = None
    publication_block: Optional[str] = None
    review_block: Optional[str] = None
    commentary_block: Optional[str] = None
    segment_warning: Optional[str] = None


class BibliographyRecord(BaseModel):
    """Final record written to JSON and flattened into CSV."""
    parser_family: str = "enumerated_robust_segments_tailclassifier_entryfix3"
    source_pdf: str
    page_start: int
    page_end: int
    section: Optional[str]
    entry_number: int
    entry_type: str
    raw_text: str
    raw_text_original: Optional[str] = None
    authors: List[str]
    year: Optional[int]
    title: Optional[str]
    publication_block: Optional[str]
    commentary: Optional[str]
    reviews: List[Review]
    confidence: float
    segment_warning: Optional[str] = None


@dataclass
class CurrentEntry:
    """Mutable accumulator for one entry while scanning PDF lines."""
    entry_number: int
    section: Optional[str]
    page_start: int
    lines: List[str]
    commentary_lines: List[str]


# -----------------------------------------------------------------------------
# Bibliographic segmentation helpers
# -----------------------------------------------------------------------------


def _squash_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_field(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = normalize_extracted_text(text)
    text = _squash_spaces(text)
    text = text.strip(" .;,:")
    return text or None



def _author_boundary_split(chunk: str) -> list[str]:
    """Split a chunk containing comma-separated non-inverted personal names.

    This preserves inverted-name commas such as "Fitzgerald, William" but splits
    cases like "M. J. Clarke, B. G. F. Currie" into two authors.
    """
    chunk = chunk.strip(" .;,:")
    if not chunk:
        return []

    # Boundary after a completed author when the next token looks like another
    # personal name beginning with initials plus a surname. This deliberately
    # does not split before plain forenames, preserving "Fitzgerald, William".
    boundary_re = re.compile(
        r",\s+(?=(?:[A-Z]\.?\s*){1,6}[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'\-]+\b)"
    )
    return [part.strip(" .;,") for part in boundary_re.split(chunk) if part.strip(" .;,")]


def _split_author_list(block: str) -> list[str]:
    """Split Werner author/editor blocks without destroying inverted names.

    Handles:
        "Fitzgerald, William and Emily Gowers"
        "M. J. Clarke, B. G. F. Currie, and R. O. A. M. Lyne"

    Keeps et-al. blocks together, since expanding them would be misleading.
    """
    block = _clean_field(block) or ""
    if not block:
        return []

    if re.search(r"\bet\s+al\.?", block, flags=re.IGNORECASE):
        return [block]

    # Normalize Oxford-comma author joins: "A, and B" -> "A and B".
    block = re.sub(r",\s+(?=(?:and|&)\s+)", " ", block, flags=re.IGNORECASE)

    parts: list[str] = []
    for piece in re.split(r"\s+(?:and|&)\s+", block):
        parts.extend(_author_boundary_split(piece))

    return [p for p in parts if p]

def _first_year(text: str) -> Optional[str]:
    m = YEAR_RE.search(text)
    return m.group(0) if m else None


def _parse_entry_line(line: str) -> Optional[Tuple[int, str]]:
    """Return (entry_number, rest) when a line starts a real enumerated entry.

    This repairs OCR-spaced numbers such as "1 5 ." -> 15 and rejects year-like
    starts such as "2006. Texte...", which are continuation lines in this corpus.
    """
    m = ENTRY_RE.match(line)
    if not m:
        return None
    digits = re.sub(r"\s+", "", m.group(1))
    if not digits:
        return None
    number = int(digits)
    if number > MAX_REASONABLE_ENTRY_NUMBER:
        return None
    return number, normalize_extracted_text(m.group(2))


def _looks_like_author_block(text: str) -> bool:
    """Conservative guard against treating publication text as authors."""
    t = _clean_field(text) or ""
    if not t:
        return False
    if len(t.split()) > 14:
        return False
    if PUB_CUE_RE.search(t):
        return False
    if YEAR_RE.search(t):
        return False
    # Allows institutional/anonymous entries like BMCR to pass elsewhere; this
    # mainly catches personal-name blocks.
    return bool(re.search(r"[A-Z][A-Za-z'\-]+", t))





def _looks_like_personal_author_block(text: str) -> bool:
    """Stricter author test for no-year ``Author. Title`` entries.

    The broad author-year fallback can safely accept loose author-ish text, but
    the ``Author. Title. Place Year`` pattern is riskier: anonymous titles,
    serials, and reference works may also have a period before their first year.
    This helper therefore requires the prefix to look like one or more personal
    names before we split on that period.
    """
    t = _clean_field(text) or ""
    if not _looks_like_author_block(t):
        return False
    if "'" in t or '"' in t:
        return False

    lowered = t.lower()
    titleish_words = ("bibliograph", "annee", "année", "gnomon", "companion", "collection")
    if any(word in lowered for word in titleish_words):
        return False

    names = _split_author_list(t)
    if not names:
        return False

    personal_name_re = re.compile(
        rf"^(?:(?:[A-Z]\.?\s*){{1,6}})?[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'\-]+"
        rf"(?:,\s*(?:(?:[A-Z]\.?\s*){{1,6}})?[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'\-]+)?"
        rf"(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'\-]+)*$"
    )

    for name in names:
        if len(name.split()) > 8:
            return False
        if not personal_name_re.match(name):
            return False
    return True


def _find_author_title_boundary(flat: str) -> Optional[int]:
    """Find the period separating a personal author prefix from a title.

    This catches records like:
        Fabio Cupaiuolo. Bibliografia della metrica latina. Napoli 1996.

    Without this pre-pass, the first-year fallback sees ``1996`` and treats
    ``Fabio Cupaiuolo. Bibliografia della metrica latina. Napoli`` as authors.
    """
    for m in re.finditer(r"\.\s+", flat):
        candidate = flat[: m.start()].strip()
        if _looks_like_personal_author_block(candidate):
            return m.start()
    return None


def _split_author_role_prefix(raw_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Split no-year entries that explicitly mark an editor/translator before title.

    This prevents the generic first-period fallback from splitting personal
    initials, e.g. treating "Craig W." as the full author.

    Returns (author_block, role, remainder_after_role, warning).
    """
    text = _squash_spaces(raw_text)
    m = AUTHOR_ROLE_PREFIX_RE.match(text)
    if not m:
        return None, None, None, None
    author = _clean_field(m.group("author"))
    role = (m.group("role") or "").lower()
    rest = _clean_field(m.group("rest"))
    if not author or not rest or not _looks_like_author_block(author):
        return None, None, None, None
    return author, role, rest, None


def _split_anonymous_quoted_article(raw_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Split anonymous/title-first quoted article entries.

    Handles entries like:
        "Bibliographische Beilage." Gnomon 78 (2006) ...

    These have no author block, so the leading quoted title must be captured
    before the generic author/title fallback sees the first period.
    """
    text = raw_text.lstrip()
    if not text.startswith('"'):
        return None, None, None

    warning = None
    after_open = text[1:]
    close_idx = after_open.find('"')
    if close_idx >= 0:
        title = after_open[:close_idx]
        publication = after_open[close_idx + 1:]
        return _clean_field(title), _clean_field(publication), warning

    # If the closing quotation mark is missing from the PDF text layer, fall
    # back to a publication cue or a visual line break, mirroring the authored
    # quoted-title logic.
    warning = "unclosed_quote_title_no_author"
    lines = after_open.splitlines()
    if len(lines) > 1 and PUB_CUE_RE.search(lines[1]):
        title = lines[0]
        publication = "\n".join(lines[1:])
        return _clean_field(title), _clean_field(publication), warning

    flat = _squash_spaces(after_open)
    cue = PUB_CUE_RE.search(flat)
    if cue and cue.start() > 0:
        title = flat[:cue.start()]
        publication = flat[cue.start():]
        return _clean_field(title), _clean_field(publication), warning

    return _clean_field(lines[0] if lines else after_open), None, warning


def _split_quoted_article(raw_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Split entries of the form:
        Author. "Title." Publication 52 (2006) 138-61.

    Important: this uses newlines before squashing spaces. If the PDF text layer
    loses a closing quotation mark, the safest title endpoint is often the end
    of the visual line, as in:
        Alexander G. McKay. "Vergilian Bibliography 200\nVergilius 52 ...
    """
    m = AUTHOR_BEFORE_QUOTE_RE.search(raw_text)
    if not m:
        return None, None, None, None

    author = _clean_field(m.group("author"))
    title_start = m.end()
    after_quote = raw_text[title_start:]

    warning = None
    title = None
    publication = None

    close_idx = after_quote.find('"')
    if close_idx >= 0:
        title = after_quote[:close_idx]
        publication = after_quote[close_idx + 1 :]
    else:
        # Missing close quote in extracted text. Use the visual line break if the
        # next line looks like a journal/publication citation; otherwise fall
        # back to the first publication cue inside the flattened remainder.
        warning = "unclosed_quote_title"
        lines = after_quote.splitlines()
        if len(lines) > 1 and PUB_CUE_RE.search(lines[1]):
            title = lines[0]
            publication = "\n".join(lines[1:])
        else:
            flat = _squash_spaces(after_quote)
            cue = PUB_CUE_RE.search(flat)
            if cue and cue.start() > 0:
                title = flat[: cue.start()]
                publication = flat[cue.start() :]
            else:
                title = lines[0] if lines else after_quote
                publication = "\n".join(lines[1:]) if len(lines) > 1 else None

    return author, _clean_field(title), _clean_field(publication), warning



# -----------------------------------------------------------------------------
# Publication-tail classifier
# -----------------------------------------------------------------------------
# Older versions depended on CITY_TOKENS. That works only until the next unseen
# place name. This version works backward from the end of an entry and asks
# whether the tail has the bibliographic *shape* of publication metadata:
#
#   Place. SIGLUM.
#   Place; Place. SIGLUM.
#   Series Name 20. Place.
#   Collection Latomus 317. Bruxelles.
#   Proceedings ... Suppl. Vol. 31. Cambridge.
#   2 vols. Roma.
#   Paris 2006.
#
# CITY_TOKENS is retained only as a last-ditch compatibility fallback, not as the
# main splitting strategy.

SIGLUM_RE = re.compile(r"^[A-ZÀ-Þ][A-ZÀ-Þ0-9&'./\-]{2,}(?:\s+[A-ZÀ-Þ][A-ZÀ-Þ0-9&'./\-]{2,}){0,5}$")

SERIES_LIKE_RE = re.compile(
    r"\b(?:"
    r"Acta|Altertumswissenschaftliches|Beihefte?|Beiträge|Beitraege|Bibliothek|"
    r"Collection|Collections|Commentar(?:ii)|Kolloquium|"
    r"Proceedings|Proc\.|Series|Studies|Studien|Supplement|Supplementum|Suppl\.|"
    r"Texte|Transactions|Vol\.|Tome|Band"
    r")\b",
    flags=re.IGNORECASE,
)

VOLUME_LIKE_RE = re.compile(r"^(?:\d+\s+vols?|Vol\.\s*\d+|Tome\s+\d+|Band\s+\d+)\.?$", flags=re.IGNORECASE)
PLACE_WITH_YEAR_RE = re.compile(r"^[A-ZÀ-Þ][A-Za-zÀ-ÿ.'’\- ]{1,40}\s+(?:19|20)\d{2}$")

_ABBREVIATION_PROTECT = {
    "Vol": "Vol<prd>",
    "vol": "vol<prd>",
    "vols": "vols<prd>",
    "Suppl": "Suppl<prd>",
    "suppl": "suppl<prd>",
    "Proc": "Proc<prd>",
    "ed": "ed<prd>",
    "eds": "eds<prd>",
    "trans": "trans<prd>",
    "Rev": "Rev<prd>",
}


def _protect_sentence_abbreviations(text: str) -> str:
    # Protect only standalone abbreviations, not words ending in the same
    # letters (e.g., do not turn "Re-Inscribed." into "Re-Inscribed<prd>").
    for literal, repl in _ABBREVIATION_PROTECT.items():
        text = re.sub(rf"(?<![A-Za-zÀ-ÿ]){re.escape(literal)}\.", repl, text)
    return text


def _restore_sentence_abbreviations(text: str) -> str:
    return text.replace("<prd>", ".")


def _sentence_chunks(text: str) -> list[str]:
    """Split into rough sentence-like chunks while protecting common bib abbrevs."""
    protected = _protect_sentence_abbreviations(_squash_spaces(text).strip())
    protected = protected.strip(" .")
    if not protected:
        return []
    chunks = [c.strip() for c in re.split(r"\.\s+", protected) if c.strip()]
    return [_restore_sentence_abbreviations(c).strip(" .") for c in chunks if c.strip(" .")]


def _join_chunks(chunks: list[str]) -> str:
    return ". ".join(c.strip(" .") for c in chunks if c.strip(" .")).strip(" .")


def _is_siglum_chunk(chunk: str) -> bool:
    return bool(SIGLUM_RE.fullmatch(chunk.strip(" .")))


def _is_series_like_chunk(chunk: str) -> bool:
    c = chunk.strip(" .")
    if not c:
        return False
    if VOLUME_LIKE_RE.fullmatch(c):
        return True
    if SERIES_LIKE_RE.search(c):
        return True
    # Generic named-series shape: a short capitalized noun phrase ending in a number.
    words = c.split()
    return bool(
        2 <= len(words) <= 10
        and re.match(r"^[A-ZÀ-Þ]", c)
        and re.search(r"\b\d+[A-Za-z]?$", c)
        and not re.search(r"[?:=]", c)
    )


def _is_place_like_chunk(chunk: str, *, allow_single_word: bool = False) -> bool:
    """Heuristic place detector for terminal publication tails.

    This deliberately does not ask whether the phrase is a known city. It asks
    whether the chunk has a compact place-like shape: title-cased words, optional
    comma/semicolon-separated region, no digits except the special "Paris 2006"
    style handled separately, and no title-ish punctuation such as ? : =.
    """
    c = chunk.strip(" .")
    if not c:
        return False
    if PLACE_WITH_YEAR_RE.fullmatch(c):
        return True
    if len(c) > 60 or re.search(r"[?:=]", c) or re.search(r"\d", c):
        return False
    if not re.match(r"^[A-ZÀ-Þ]", c):
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ.'’\-]*", c)
    if not words or len(words) > 6:
        return False
    has_place_separator = bool(re.search(r"[,;]", c))
    if len(words) == 1 and not (allow_single_word or has_place_separator):
        return False
    # Compact title-case / abbreviation tokens. This allows "New York",
    # "Swansea, Wales", "Oxford; New York", and "Cambridge, MA".
    for word in words:
        if len(word) <= 3 and word.isupper():
            continue
        if not re.match(r"^[A-ZÀ-Þ][A-Za-zÀ-ÿ.'’\-]*$", word):
            return False
    return True


def _split_at_publication_start_cue(remainder: str) -> Tuple[Optional[str], Optional[str]]:
    """Split at explicit publication-series openings, if present."""
    for m in PUBLICATION_START_RE.finditer(remainder):
        if m.start() <= 8:
            continue
        title = remainder[: m.start()]
        pub = remainder[m.start() + 2 :]
        return _clean_field(title), _clean_field(pub)
    return None, None


def _split_by_publication_tail(remainder: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Split title/publication using terminal-tail shape rather than city names.

    Returns (title, publication_block, warning). A warning is provided when the
    split is inferred from a generic place-like tail rather than an explicit
    series/volume/siglum cue.
    """
    remainder = _squash_spaces(remainder)
    if not remainder:
        return None, None, None

    title, pub = _split_at_publication_start_cue(remainder)
    if title and pub:
        return title, pub, None

    chunks = _sentence_chunks(remainder)
    if len(chunks) < 2:
        return _clean_field(remainder), None, None

    idx = len(chunks) - 1
    pub_start = None
    warning = None

    # 1. Optional terminal all-caps/siglum chunk(s), e.g. BOOTH/MALTBY.
    while idx >= 0 and _is_siglum_chunk(chunks[idx]):
        pub_start = idx
        idx -= 1

    has_siglum = pub_start is not None

    if has_siglum:
        # If a siglum is present, the publication tail usually begins at the
        # nearest preceding place-like chunk, and may include short bracketed
        # series/cross-reference chunks between that place and the siglum:
        #   Oxford. [Blackwell Companions...]. DOMINIK/HALL.
        #   Swansea, Wales. BOOTH/MALTBY.
        place_idx = None
        for j in range(idx, max(-1, idx - 6), -1):
            if _is_place_like_chunk(chunks[j], allow_single_word=True):
                place_idx = j
                break
        if place_idx is not None:
            pub_start = place_idx
            idx = place_idx - 1

    # 2. Terminal place-ish chunk immediately before siglum or at absolute end.
    # Single-word unknown places are accepted when followed by a siglum or
    # preceded by a series/volume chunk; otherwise they are treated cautiously.
    if not has_siglum and idx >= 0:
        prev_is_series = idx > 0 and _is_series_like_chunk(chunks[idx - 1])
        # Allow a final single title-case word as a publication-place tail, but
        # mark it as inferred unless it is anchored by a preceding series/volume
        # cue or a comma/semicolon place phrase. This replaces the old behavior
        # of continually adding "Heidelberg", "Exeter", "Columbus", etc. to a
        # city list.
        allow_single = True
        if _is_place_like_chunk(chunks[idx], allow_single_word=allow_single):
            pub_start = idx
            idx -= 1
            if not (prev_is_series or re.search(r"[,;]", chunks[pub_start]) or PLACE_WITH_YEAR_RE.fullmatch(chunks[pub_start])):
                warning = "publication_tail_inferred"

    # 3. Pull preceding numbered series / volume-count chunks into pub block.
    while pub_start is not None and idx >= 0 and _is_series_like_chunk(chunks[idx]):
        pub_start = idx
        idx -= 1

    if pub_start is not None and pub_start > 0:
        return _clean_field(_join_chunks(chunks[:pub_start])), _clean_field(_join_chunks(chunks[pub_start:])), warning

    return _clean_field(remainder), None, None


# Compatibility fallback retained for difficult cases. It is deliberately used
# only after the publication-tail classifier has failed, so new place names do
# not require expanding CITY_TOKENS for ordinary cases.
def _city_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for token in sorted(CITY_TOKENS, key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(token) + r"\b", text):
            spans.append((m.start(), m.end(), token))
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept: list[tuple[int, int, str]] = []
    occupied: list[range] = []
    for start, end, token in spans:
        if any(start < r.stop and end > r.start for r in occupied):
            continue
        kept.append((start, end, token))
        occupied.append(range(start, end))
    return sorted(kept, key=lambda x: x[0])


def _city_tail_is_publication_boundary(text: str, city_end: int) -> bool:
    tail = text[city_end:].strip(" .;:")
    if not tail:
        return True
    return bool(SIGLUM_RE.fullmatch(tail))


def _expand_city_tail_start(text: str, terminal_city_start: int) -> int:
    spans = _city_spans(text[:terminal_city_start])
    start = terminal_city_start
    changed = True
    while changed:
        changed = False
        for prev_start, prev_end, _token in reversed(spans):
            if prev_start >= start:
                continue
            between = text[prev_end:start]
            if not re.fullmatch(r"\s*[;,]\s*", between):
                continue
            before = text[:prev_start].rstrip()
            if before and not before.endswith("."):
                continue
            start = prev_start
            changed = True
            break
    return start


def _split_by_city_compatibility_fallback(remainder: str) -> Tuple[Optional[str], Optional[str]]:
    remainder = _squash_spaces(remainder)
    spans = _city_spans(remainder)
    if not spans:
        return None, None
    stripped = remainder.rstrip(" .;," )
    terminal_spans = [(start, end, token) for start, end, token in spans if _city_tail_is_publication_boundary(stripped, end)]
    if terminal_spans:
        idx = _expand_city_tail_start(remainder, terminal_spans[-1][0])
    else:
        idx = spans[0][0]
    if idx <= 8:
        return None, None
    return _clean_field(remainder[:idx]), _clean_field(remainder[idx:])


def _split_by_city_or_year(remainder: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Fallback split for monographs / edited volumes without quoted titles.

    The primary strategy is now a terminal publication-tail classifier. It works
    from entry shape rather than from a city/place vocabulary. A small city-token
    fallback is retained for backward compatibility after the shape-based method
    fails.
    """
    remainder = _squash_spaces(remainder)
    if not remainder:
        return None, None, None

    title, pub, warning = _split_by_publication_tail(remainder)
    if pub:
        return title, pub, warning

    title, pub = _split_by_city_compatibility_fallback(remainder)
    if pub:
        return title, pub, "publication_tail_city_fallback"

    years = list(YEAR_RE.finditer(remainder))
    if years:
        y = years[-1]
        if y.start() > 8:
            return _clean_field(remainder[: y.start()]), _clean_field(remainder[y.start() :]), "publication_tail_year_fallback"

    return _clean_field(remainder), None, None

class EnumeratedParser:
    """Parser for numbered Werner bibliography PDFs.

    One instance owns an output directory and writes both per-PDF JSON files and
    optional CSV output. The parser assumes entries are enumerated; it is not the
    right parser for later unnumbered/hanging-indent Werner volumes.
    """
    def __init__(self, output_dir: Path):
        """Create output folders used by parse_pdf()."""
        self.output_dir = output_dir
        self.record_dir = output_dir / "parsed_records"
        self.record_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """Remove page-level boilerplate before line-by-line entry detection."""
        text = normalize_extracted_text(text)
        # Common JSTOR boilerplate / access metadata.  These sometimes appear
        # as standalone lines rather than as part of the longer boilerplate
        # paragraphs, so remove both families here and again at line level.
        text = re.sub(r"This content downloaded from.*", "", text)
        text = re.sub(r"All use subject to.*", "", text)
        text = re.sub(r"JSTOR is a not-for-profit service.*", "", text)
        text = re.sub(r"^\s*\d{1,3}(?:\.\d{1,3}){3}\s+on\s+.*UTC\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\s+Shirley\s+Werner\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*Shirley\s+Werner\s+\d+\s*$", "", text, flags=re.MULTILINE)
        return normalize_extracted_text(text)

    def is_noise_line(self, line: str) -> bool:
        """Detect standalone JSTOR/access/page-footer lines that survived cleanup."""
        line = normalize_extracted_text(line).strip()
        if not line:
            return True
        if re.match(r"^\d{1,3}(?:\.\d{1,3}){3}\s+on\s+.*UTC$", line):
            return True
        if re.match(r"^\d+\s+Shirley\s+Werner$", line):
            return True
        if re.match(r"^Shirley\s+Werner\s+\d+$", line):
            return True
        if line.startswith("This content downloaded from"):
            return True
        if line.startswith("All use subject to"):
            return True
        if line.startswith("JSTOR is a not-for-profit service"):
            return True
        if RUNNING_HEADER_RE.match(line):
            return True
        return False

    def is_section_header(self, line: str) -> bool:
        """Recognize all-caps section headers such as `BIBLIOGRAPHY`."""
        line = line.strip()
        if len(line.split()) > 8:
            return False
        if RUNNING_HEADER_RE.match(line):
            return False
        return bool(SECTION_RE.match(line))

    def is_commentary(self, line: str) -> bool:
        """Conservatively classify long non-entry lines as annotation/commentary."""
        if _parse_entry_line(line) is not None:
            return False
        if REVIEW_RE.search(line):
            return False
        return len(line.split()) > 24

    def segment_entry(self, raw_text: str, commentary: Optional[str] = None) -> BibliographicSegments:
        """Split one normalized entry into author/year/title/publication/review fields.

        The ordering of cases is deliberate: specific high-confidence structures
        run first, and the generic author-year fallback runs last. This prevents
        initials, anonymous quoted titles, or editor markers from being mistaken
        for author/title boundaries.
        """
        raw_text_original = normalize_extracted_text(raw_text)
        raw_text = normalize_entry_text(raw_text_original)
        if commentary is not None:
            commentary = normalize_entry_text(commentary)

        review_block = None
        working = raw_text
        review_match = REVIEW_RE.search(working)
        if review_match:
            review_block = _clean_field(review_match.group(1))
            working = working[: review_match.start()].strip()

        # First preference: quoted-title article segmentation. This is much more
        # reliable than splitting at the first year, because journal articles
        # often have no author-date year immediately after the author.
        author, title, publication, warning = _split_quoted_article(working)
        if author or title or publication:
            year = _first_year(publication or working)
            return BibliographicSegments(
                raw_text=raw_text,
                author_block=author,
                year_block=year,
                title_block=title,
                publication_block=publication,
                review_block=review_block,
                commentary_block=commentary,
                segment_warning=warning,
            )

        # Anonymous/title-first quoted article entries, e.g.:
        #   "Bibliographische Beilage." Gnomon 78 (2006) ...
        # There is no author block to split before the quote.
        title, publication, warning = _split_anonymous_quoted_article(working)
        if title or publication:
            year = _first_year(publication or working)
            return BibliographicSegments(
                raw_text=raw_text,
                author_block=None,
                year_block=year,
                title_block=title,
                publication_block=publication,
                review_block=review_block,
                commentary_block=commentary,
                segment_warning=warning,
            )

        # Explicit no-year role prefix, e.g.:
        #   Craig W. Kallendorf, ed. A Companion to the Classical Tradition
        # Must run before the generic first-period fallback so initials are not
        # mistaken for the author/title separator.
        author, role, role_remainder, role_warning = _split_author_role_prefix(working)
        if author and role_remainder:
            role_remainder_working = role_remainder
            role_year = None
            role_year_match = re.match(r"^((?:19|20)\d{2})\.?,?\s+(.+)$", role_remainder_working)
            if role_year_match:
                role_year = role_year_match.group(1)
                role_remainder_working = role_year_match.group(2)
            title_block, publication_block, tail_warning = _split_by_city_or_year(role_remainder_working)
            return BibliographicSegments(
                raw_text=raw_text,
                author_block=author,
                year_block=role_year or _first_year(publication_block or role_remainder_working),
                title_block=title_block,
                publication_block=publication_block,
                review_block=review_block,
                commentary_block=commentary,
                segment_warning=tail_warning or role_warning,
            )

        flat = _squash_spaces(working)

        # Fallback A: personal-author prefix entries without author-date order.
        #
        # Some records have the form:
        #     Fabio Cupaiuolo. Bibliografia della metrica latina. Napoli 1996.
        # Here the first year belongs to the publication tail, not to an
        # author-date pattern. If we run the generic year fallback first, it
        # mistakenly treats "Fabio Cupaiuolo. Bibliografia... Napoli" as the
        # author block. So we first look for a stricter personal-name prefix
        # followed by a period and title.
        author_block = None
        year_block = None
        remainder = flat

        author_title_boundary = _find_author_title_boundary(flat)
        if author_title_boundary is not None:
            author_block = flat[:author_title_boundary]
            remainder = flat[author_title_boundary + 2 :]
            year_block = _first_year(remainder)

        # Fallback B: author-date-ish entries.
        year_match = YEAR_RE.search(flat)
        if author_block is None and year_match:
            possible_author = flat[: year_match.start()].strip(" .;,")
            # If the prefix already contains a sentence boundary, the year is
            # probably in the publication tail rather than author-date position.
            if _looks_like_author_block(possible_author) and ". " not in possible_author:
                author_block = possible_author
                year_block = year_match.group(0)
                remainder = flat[year_match.end() :].strip()

        # Fallback C: broad Author. Title... entries without an author-date year.
        if author_block is None:
            dot = flat.find(". ")
            if dot > 0:
                possible_author = flat[:dot]
                if _looks_like_author_block(possible_author):
                    author_block = possible_author
                    remainder = flat[dot + 2 :]
                    year_block = _first_year(remainder)

        title_block, publication_block, tail_warning = _split_by_city_or_year(remainder)
        if year_block is None:
            year_block = _first_year(publication_block or remainder)

        return BibliographicSegments(
            raw_text=raw_text,
            author_block=_clean_field(author_block),
            year_block=year_block,
            title_block=title_block,
            publication_block=publication_block,
            review_block=review_block,
            commentary_block=commentary,
            segment_warning=tail_warning,
        )

    def classify_entry_type(self, segments: BibliographicSegments) -> EntryType:
        """Infer a coarse entry type from role markers and publication cues."""
        text = segments.raw_text.lower()
        if "diss." in text or "dissertation" in text:
            return EntryType.DISSERTATION
        if "trans." in text or " translated " in text:
            return EntryType.TRANSLATION
        if re.search(r"\b(?:eds?|editor|editors)\.", text):
            return EntryType.EDITED_VOLUME
        pub = segments.publication_block or ""
        # Series blocks such as "Collection Latomus 317. Bruxelles" are
        # monograph/volume metadata, not journal-article citations.
        if re.search(r"^Collection\s+", pub, flags=re.IGNORECASE):
            return EntryType.MONOGRAPH
        if '"' in segments.raw_text or (pub and PUB_CUE_RE.search(pub) and re.search(r"\b\d+\s*\(", pub)):
            return EntryType.JOURNAL_ARTICLE
        if segments.publication_block:
            return EntryType.MONOGRAPH
        return EntryType.UNKNOWN

    def extract_authors(self, segments: BibliographicSegments) -> List[str]:
        """Normalize and split the author/editor block into a list of names."""
        if not segments.author_block:
            return []

        block = normalize_for_parsing(segments.author_block)
        # Remove terminal role markers without leaving fragments such as "s" from
        # "eds.". Keep non-terminal words intact.
        block = re.sub(r",?\s*\b(?:eds?|trans)\.?,?\s*$", "", block, flags=re.IGNORECASE)
        block = _clean_field(block) or ""
        if not block:
            return []

        if ";" in block:
            authors = []
            for piece in block.split(";"):
                authors.extend(_split_author_list(piece))
        else:
            authors = _split_author_list(block)

        # Filter obviously overlong accidental blocks, but keep institutional names.
        return [a for a in authors if len(a.split()) <= 12]

    def extract_title(self, segments: BibliographicSegments) -> Optional[str]:
        """Return a cleaned title, or None if the candidate is clearly empty/noisy."""
        title = _clean_field(segments.title_block)
        if not title or len(title) < 4:
            return None
        return title

    def finalize_entry(self, entry: CurrentEntry, page_end: int, source_pdf: str) -> BibliographyRecord:
        """Convert an accumulated CurrentEntry into the final output record.

        This is where audit text, parsed fields, entry type, confidence, and
        warnings are assembled into one object.
        """
        raw_text_original = "\n".join(entry.lines)
        raw_text = normalize_entry_text(raw_text_original)
        commentary_original = "\n".join(entry.commentary_lines) if entry.commentary_lines else None
        commentary = normalize_entry_text(commentary_original) if commentary_original else None

        segments = self.segment_entry(raw_text_original, commentary_original)
        authors = self.extract_authors(segments)
        title = self.extract_title(segments)

        year = int(segments.year_block) if segments.year_block else None

        confidence = 0.0
        if authors:
            confidence += 0.3
        if year:
            confidence += 0.25
        if title:
            confidence += 0.35
        if segments.publication_block:
            confidence += 0.1
        if segments.segment_warning:
            confidence -= 0.15
        confidence = max(0.0, min(1.0, confidence))

        reviews: list[Review] = []
        if segments.review_block:
            reviews.append(Review(raw_text=segments.review_block))

        return BibliographyRecord(
            source_pdf=source_pdf,
            page_start=entry.page_start,
            page_end=page_end,
            section=entry.section,
            entry_number=entry.entry_number,
            entry_type=self.classify_entry_type(segments).value,
            raw_text=raw_text,
            raw_text_original=normalize_extracted_text(raw_text_original),
            authors=authors,
            year=year,
            title=title,
            publication_block=_clean_field(segments.publication_block),
            commentary=commentary,
            reviews=reviews,
            confidence=round(confidence, 2),
            segment_warning=segments.segment_warning,
        )

    def _is_commentary_prose_line(self, line: str) -> bool:
        """Return True when a line looks like the start of a group commentary paragraph.

        McKay (and Werner) bibliography sections sometimes follow a group of
        numbered entries with a prose paragraph that discusses several of those
        entries at once.  These paragraphs must NOT be attached to the last
        numbered entry — they belong to the group as a whole.

        Three detection patterns are used:

        1. Possessive personal name: "Lyne's introduction takes a fresh..."
        2. Surname + scholarly verb within first six words:
           "Geymonat favours Bucolica", "Cicu analyses Georgics"
        3. Early parenthetical entry reference + prose density:
           "Poeschi (1) offers a clear analysis" (McKay's notation style)
        """
        line = line.strip()
        if not line:
            return False
        if _parse_entry_line(line) is not None:
            return False
        if REVIEW_RE.search(line):
            return False
        if self.is_section_header(line):
            return False
        if line.startswith('"') or line.startswith("'"):
            return False

        words = line.split()
        if len(words) < 5:
            return False

        # Pattern 1: possessive personal surname, e.g. "Lyne's introduction…"
        if re.match(r"^[A-ZÀ-Þ][a-zÀ-ÿ]{2,}\'s\s+", line):
            return True

        # Pattern 2: capitalized surname + scholarly verb within first six words
        first_word_alpha = re.sub(r"[^A-Za-zÀ-ÿ]", "", words[0])
        if re.match(r"^[A-ZÀ-Þ][a-zÀ-ÿ]{2,}", first_word_alpha):
            for w in words[1:6]:
                if re.sub(r"[^a-zA-Z]", "", w).lower() in _COMMENTARY_VERBS:
                    return True

        # Pattern 3: parenthetical entry reference in first five words
        # (McKay's commentary often references its own entries as "(N)")
        early_text = " ".join(words[:5])
        if re.search(r"\(\d+\)", early_text) and len(words) >= 8:
            lowercase_count = sum(1 for w in words if w and w[0].islower())
            if lowercase_count >= 4:
                return True

        return False

    def _make_commentary_record(
        self,
        lines: List[str],
        page_start: int,
        page_end: int,
        section: Optional[str],
        preceding_entry_number: int,
        source_pdf: str,
    ) -> "BibliographyRecord":
        """Build a BibliographyRecord for a group commentary block."""
        text_original = "\n".join(lines)
        text = normalize_entry_text(text_original)
        return BibliographyRecord(
            parser_family="group_commentary",
            source_pdf=source_pdf,
            page_start=page_start,
            page_end=page_end,
            section=section,
            entry_number=preceding_entry_number,
            entry_type="group_commentary",
            raw_text=text,
            raw_text_original=normalize_extracted_text(text_original),
            authors=[],
            year=None,
            title=None,
            publication_block=None,
            commentary=text,
            reviews=[],
            confidence=1.0,
        )

    def parse_pdf(self, pdf_path: Path) -> List[BibliographyRecord]:
        """Parse one PDF into records and write its per-PDF JSON file."""
        doc = fitz.open(pdf_path)
        current_section = None
        current_entry: Optional[CurrentEntry] = None
        records: list[BibliographyRecord] = []

        # Group commentary state — a prose paragraph that follows a block of
        # numbered entries and discusses multiple of them at once.  It must be
        # kept separate from any single entry's data.
        commentary_lines: list[str] = []
        commentary_page_start: int = 1
        last_entry_number: int = 0  # tracks the last finalized entry number

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

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            text = self.clean_text(page.get_text("text"))
            lines = []
            for raw_line in text.splitlines():
                line = normalize_extracted_text(raw_line).strip()
                if line and not self.is_noise_line(line):
                    lines.append(line)

            for line in lines:
                if self.is_section_header(line):
                    flush_commentary(page_number + 1)
                    current_section = line
                    continue

                entry_info = _parse_entry_line(line)
                if entry_info is not None:
                    flush_commentary(page_number + 1)
                    entry_number, entry_text = entry_info
                    if current_entry:
                        records.append(self.finalize_entry(current_entry, page_number + 1, pdf_path.name))
                        last_entry_number = current_entry.entry_number
                        current_entry = None
                    current_entry = CurrentEntry(
                        entry_number=entry_number,
                        section=current_section,
                        page_start=page_number + 1,
                        lines=[entry_text],
                        commentary_lines=[],
                    )
                    continue

                # Skip pre-content lines before any entry has been seen
                if current_entry is None and not commentary_lines:
                    continue

                # Detect the start of a new group commentary block.
                # Once detected, the preceding entry is finalised immediately so
                # the commentary text is never mixed into a single entry's data.
                if not commentary_lines and self._is_commentary_prose_line(line):
                    if current_entry:
                        records.append(self.finalize_entry(current_entry, page_number + 1, pdf_path.name))
                        last_entry_number = current_entry.entry_number
                        current_entry = None
                    commentary_page_start = page_number + 1
                    commentary_lines.append(line)
                    continue

                if commentary_lines:
                    # Continue accumulating the commentary block until a new
                    # entry or section header resets the state (handled above).
                    commentary_lines.append(line)
                    continue

                # Normal entry continuation line
                if self.is_commentary(line):
                    current_entry.commentary_lines.append(line)
                else:
                    current_entry.lines.append(line)

        flush_commentary(len(doc))
        if current_entry:
            records.append(self.finalize_entry(current_entry, len(doc), pdf_path.name))

        output_path = self.record_dir / f"{pdf_path.stem}.json"
        output_path.write_text(
            json.dumps([r.model_dump() for r in records], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return records


# -----------------------------------------------------------------------------
# CSV output helpers
# -----------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "parser_family",
    "source_pdf",
    "page_start",
    "page_end",
    "section",
    "entry_number",
    "entry_type",
    "authors",
    "year",
    "title",
    "publication_block",
    "commentary",
    "reviews",
    "confidence",
    "segment_warning",
    "raw_text",
    "raw_text_original",
]


def record_to_csv_row(record: BibliographyRecord) -> dict[str, object]:
    """Flatten a BibliographyRecord into a spreadsheet-friendly CSV row.

    JSON remains the richer archival format. The CSV uses pipe-separated values
    for naturally repeated fields such as authors and reviews.
    """
    data = record.model_dump()
    return {
        "parser_family": data.get("parser_family"),
        "source_pdf": data.get("source_pdf"),
        "page_start": data.get("page_start"),
        "page_end": data.get("page_end"),
        "section": data.get("section") or "",
        "entry_number": data.get("entry_number"),
        "entry_type": data.get("entry_type"),
        "authors": " | ".join(data.get("authors") or []),
        "year": data.get("year") or "",
        "title": data.get("title") or "",
        "publication_block": data.get("publication_block") or "",
        "commentary": data.get("commentary") or "",
        "reviews": " | ".join(r.get("raw_text", "") for r in (data.get("reviews") or [])),
        "confidence": data.get("confidence"),
        "segment_warning": data.get("segment_warning") or "",
        "raw_text": data.get("raw_text") or "",
        "raw_text_original": data.get("raw_text_original") or "",
    }


def write_records_csv(records: list[BibliographyRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record_to_csv_row(record))


# -----------------------------------------------------------------------------
# Command-line interface and built-in regression/self tests
# -----------------------------------------------------------------------------
# The self-test flags exercise specific bugs found during corpus inspection.
# They are intentionally small, example-driven regression tests rather than a
# full unit-test suite. Running the script without self-test flags parses all
# PDFs in the supplied directory and writes JSON/CSV outputs.

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--csv-name", default="enumerated_records.csv", help="Name of the combined CSV written to output_dir.")
    parser.add_argument("--per-pdf-csv", action="store_true", help="Also write one CSV per PDF next to the per-PDF JSON files.")
    parser.add_argument("--encoding-self-test", action="store_true")
    parser.add_argument("--dehyphenation-self-test", action="store_true")
    parser.add_argument("--pubblock-self-test", action="store_true")
    parser.add_argument("--series-self-test", action="store_true")
    parser.add_argument("--generic-series-self-test", action="store_true")
    parser.add_argument("--raw-text-normalization-self-test", action="store_true")
    parser.add_argument("--pubtail-self-test", action="store_true")
    parser.add_argument("--author-pubtail-self-test", action="store_true")
    parser.add_argument("--remshalden-pubtail-self-test", action="store_true")
    parser.add_argument("--anonymous-quote-self-test", action="store_true")
    parser.add_argument(
        "--segment-self-test",
        action="store_true",
        help="Print segmentation for the known Werner 53 entry #2 failure case.",
    )

    args = parser.parse_args()

    if args.encoding_self_test:
        sample = "m√©moire et √©chos"
        console.print(f"Encoding repair test: {sample!r} -> {repair_mojibake(sample)!r}")

    if args.segment_self_test:
        engine = EnumeratedParser(args.output_dir)
        sample = 'Alexander G. McKay. "Vergilian Bibliography 200\nVergilius 52 (2006) 138-61.'
        seg = engine.segment_entry(sample)
        console.print(seg.model_dump())

    if args.dehyphenation_self_test:
        engine = EnumeratedParser(args.output_dir)
        sample = "Arduini, P. et al., eds. 2008. Studi offerti ad Alessandro Pe-\nrutelli. 2 vols. Roma."
        console.print(f"Dehyphenation test: {dehyphenate_pdf_linebreaks(sample)!r}")
        console.print(engine.segment_entry(sample).model_dump())
        rec = CurrentEntry(
            entry_number=4,
            section="LEXICA, COMPANIONS, AND COLLECTIONS",
            page_start=3,
            lines=sample.splitlines(),
            commentary_lines=[],
        )
        console.print(engine.finalize_entry(rec, 3, "55 (2009) - Werner.pdf").model_dump())

    if args.pubblock_self_test:
        engine = EnumeratedParser(args.output_dir)
        sample = "Fitzgerald, William and Emily Gowers, eds. 2007. Ennius peren-\nnis : The Annals and Beyond. Proceedings of the Cambridge\nPhilological Society Suppl. Vol. 31. Cambridge."
        rec = CurrentEntry(
            entry_number=7,
            section="LEXICA, COMPANIONS, AND COLLECTIONS",
            page_start=3,
            lines=sample.splitlines(),
            commentary_lines=[],
        )
        console.print(engine.finalize_entry(rec, 3, "55 (2009) - Werner.pdf").model_dump())


    if args.series_self_test:
        engine = EnumeratedParser(args.output_dir)
        sample = "Casali, Sergio and Fabio Stok. 2008. Servio : stratificazioni\nesegetiche e modelli culturali = Servius: Exegetical Stratifica-\ntions and Cultural Models. Collection Latomus 317. Bruxelles."
        rec = CurrentEntry(
            entry_number=6,
            section="LEXICA, COMPANIONS, AND COLLECTIONS",
            page_start=3,
            lines=sample.splitlines(),
            commentary_lines=[],
        )
        console.print(engine.finalize_entry(rec, 3, "55 (2009) - Werner.pdf").model_dump())


    if args.generic_series_self_test:
        engine = EnumeratedParser(args.output_dir)
        sample = "Freund, Stefan, Meinolf Vielberg, et al., eds. 2008. Vergil und\ndas antike Epos : Festschrift Hans Jürgen Tschiedel. Altertums-\nwissenschaftliches Kolloquium 20. Stuttgart."
        rec = CurrentEntry(
            entry_number=8,
            section="LEXICA, COMPANIONS, AND COLLECTIONS",
            page_start=3,
            lines=sample.splitlines(),
            commentary_lines=[],
        )
        console.print(engine.finalize_entry(rec, 3, "55 (2009) - Werner.pdf").model_dump())

    if args.pubtail_self_test:
        engine = EnumeratedParser(args.output_dir)
        sample = "Joan Booth and Robert Maltby, eds. 2006. What 's in a Name? Th\nSignificance of Proper Names in Classical Latin Literature.\nSwansea, Wales. BOOTH/MALTBY."
        rec = CurrentEntry(
            entry_number=8,
            section="LEXICA, COMPANIONS, AND COLLECTIONS",
            page_start=3,
            lines=sample.splitlines(),
            commentary_lines=[],
        )
        console.print(engine.finalize_entry(rec, 3, "53 (2007) - Werner.pdf").model_dump())


    if args.author_pubtail_self_test:
        engine = EnumeratedParser(args.output_dir)
        sample = "M. J. Clarke, В. G. F. Currie, and R. О. A. M. Lyne, eds. 2006.\nEpic Interactions: Perspectives on Homer, Virgil, and the Epic\nTradition: Presented to Jasper Griffin by Former Pupils. Oxford;\nNew York. CLARKE/CURRIE/LYNE. Rev: S. Goldhill, BMCR\n2007.06.17."
        rec = CurrentEntry(
            entry_number=10,
            section="LEXICA, COMPANIONS, AND COLLECTIONS",
            page_start=3,
            lines=sample.splitlines(),
            commentary_lines=[],
        )
        console.print(engine.finalize_entry(rec, 3, "53 (2007) - Werner.pdf").model_dump())


    if args.remshalden_pubtail_self_test:
        engine = EnumeratedParser(args.output_dir)
        sample = "Robert Bedon and Michel Polfer, eds. 2007. Être romain.\nHommages in memoriam Charles Marie Ternes. Remshalden.\nBEDON/POLFER."
        rec = CurrentEntry(
            entry_number=7,
            section="LEXICA, COMPANIONS, AND COLLECTIONS",
            page_start=3,
            lines=sample.splitlines(),
            commentary_lines=[],
        )
        console.print(engine.finalize_entry(rec, 3, "53 (2007) - Werner.pdf").model_dump())



    if args.anonymous_quote_self_test:
        engine = EnumeratedParser(args.output_dir)
        sample = '"Bibliographische Beilage." Gnomon 78 (200'
        rec = CurrentEntry(
            entry_number=5,
            section="BIBLIOGRAPHY",
            page_start=3,
            lines=sample.splitlines(),
            commentary_lines=[],
        )
        console.print(engine.finalize_entry(rec, 3, "53 (2007) - Werner.pdf").model_dump())

    parser_engine = EnumeratedParser(args.output_dir)
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))

    all_records: list[BibliographyRecord] = []

    total = 0
    for pdf in track(pdfs, description="Parsing"):
        records = parser_engine.parse_pdf(pdf)
        all_records.extend(records)
        total += len(records)

        if args.per_pdf_csv:
            per_pdf_csv_path = parser_engine.record_dir / f"{pdf.stem}.csv"
            write_records_csv(records, per_pdf_csv_path)

        console.print(f"Parsed {pdf.name}: {len(records)} records")

    combined_csv_path = args.output_dir / args.csv_name
    write_records_csv(all_records, combined_csv_path)

    console.print(f"\nTotal extracted records: {total}")
    console.print(f"Combined CSV written to: {combined_csv_path}")
    if args.per_pdf_csv:
        console.print(f"Per-PDF CSV files written to: {parser_engine.record_dir}")
