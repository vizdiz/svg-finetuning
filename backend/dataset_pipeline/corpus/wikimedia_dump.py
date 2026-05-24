from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
import math
import re
import shutil
from typing import Iterator
from urllib.parse import quote

from backend.dataset_pipeline.config import ScrapeConfig


_SELECTED_IMAGE_COLUMNS = {0, 1, 2, 3, 6, 7, 8, 11, 12}

_POSITIVE_FILENAME_TOKENS = (
    "diagram",
    "flow",
    "chart",
    "graph",
    "network",
    "architecture",
    "pipeline",
    "schema",
    "schematic",
    "block",
    "system",
    "process",
    "sequence",
    "circuit",
    "map",
    "timeline",
    "tree",
    "uml",
    "topology",
    "logic",
)

_NEGATIVE_FILENAME_TOKENS = (
    "logo",
    "flag",
    "icon",
    "badge",
    "emblem",
    "seal",
    "favicon",
    "wordmark",
    "crest",
)


@dataclass(frozen=True, slots=True)
class WikimediaDumpCandidate:
    filename: str
    size_bytes: int
    width: int
    height: int
    media_type: str
    major_mime: str
    minor_mime: str
    timestamp: str
    sha1: str


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _ensure_decompressed_dump(dump_path: Path) -> Path:
    if dump_path.suffix != ".gz":
        return dump_path
    decompressed_path = dump_path.with_suffix("")
    if decompressed_path.exists() and decompressed_path.stat().st_mtime >= dump_path.stat().st_mtime:
        return decompressed_path
    tmp_path = decompressed_path.with_suffix(decompressed_path.suffix + ".tmp")
    with gzip.open(dump_path, "rb") as src, tmp_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    tmp_path.replace(decompressed_path)
    return decompressed_path


def _open_dump(path: str):
    dump_path = Path(path)
    if not dump_path.exists():
        raise FileNotFoundError(f"Wikimedia dump not found: {dump_path}")
    if dump_path.suffix == ".gz":
        return _ensure_decompressed_dump(dump_path).open("rt", encoding="utf-8", errors="replace")
    return dump_path.open("rt", encoding="utf-8", errors="replace")


def _parse_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def iter_insert_rows(line: str) -> Iterator[dict[int, str | None]]:
    values_at = line.find(" VALUES ")
    if values_at < 0:
        return

    i = values_at + len(" VALUES ")
    n = len(line)
    while i < n:
        while i < n and line[i] in " \t\r\n,;":
            i += 1
        if i >= n or line[i] != "(":
            break
        i += 1
        field_index = 0
        row: dict[int, str | None] = {}
        while i < n:
            collect = field_index in _SELECTED_IMAGE_COLUMNS
            token: list[str] = []
            if line[i] == "'":
                i += 1
                while i < n:
                    ch = line[i]
                    if ch == "\\" and i + 1 < n:
                        if collect:
                            token.append(line[i + 1])
                        i += 2
                        continue
                    if ch == "'":
                        i += 1
                        break
                    if collect:
                        token.append(ch)
                    i += 1
                value: str | None = "".join(token) if collect else None
            else:
                start = i
                while i < n and line[i] not in ",)":
                    i += 1
                raw = line[start:i].strip()
                value = None if raw.upper() == "NULL" else raw
            if collect:
                row[field_index] = value
            while i < n and line[i].isspace():
                i += 1
            if i >= n:
                break
            if line[i] == ",":
                field_index += 1
                i += 1
                continue
            if line[i] == ")":
                i += 1
                yield row
                break
            i += 1


def candidate_from_image_row(row: dict[int, str | None]) -> WikimediaDumpCandidate | None:
    filename = row.get(0) or ""
    if not filename.lower().endswith(".svg"):
        return None
    candidate = WikimediaDumpCandidate(
        filename=filename,
        size_bytes=_parse_int(row.get(1)),
        width=_parse_int(row.get(2)),
        height=_parse_int(row.get(3)),
        media_type=row.get(6) or "",
        major_mime=row.get(7) or "",
        minor_mime=row.get(8) or "",
        timestamp=row.get(11) or "",
        sha1=row.get(12) or "",
    )
    if candidate.major_mime != "image" or candidate.minor_mime not in {"svg+xml", "svg"}:
        return None
    return candidate


def passes_config(candidate: WikimediaDumpCandidate, config: ScrapeConfig) -> bool:
    if candidate.size_bytes < config.min_svg_bytes or candidate.size_bytes > config.max_svg_bytes:
        return False
    if candidate.width <= 0 or candidate.height <= 0:
        return False
    aspect_ratio = candidate.width / candidate.height
    return config.hard_min_aspect_ratio <= aspect_ratio <= config.hard_max_aspect_ratio


def iter_dump_candidates(path: str, config: ScrapeConfig) -> Iterator[WikimediaDumpCandidate]:
    with _open_dump(path) as fh:
        for line in fh:
            if not line.startswith("INSERT INTO `image` VALUES "):
                continue
            for row in iter_insert_rows(line):
                candidate = candidate_from_image_row(row)
                if candidate is not None and passes_config(candidate, config):
                    yield candidate


def score_dump_candidate(candidate: WikimediaDumpCandidate) -> float:
    filename = candidate.filename.lower()
    stem = filename.rsplit(".", 1)[0]
    score = 0.0
    for token in _POSITIVE_FILENAME_TOKENS:
        if token in stem:
            score += 18.0
    for token in _NEGATIVE_FILENAME_TOKENS:
        if token in stem:
            score -= 35.0
    area = candidate.width * candidate.height
    score += min(35.0, max(0.0, math.log10(max(area, 1)) * 8.0))
    score += min(15.0, max(0.0, math.log10(max(candidate.size_bytes, 1)) * 4.0))
    aspect_ratio = candidate.width / candidate.height
    aspect_penalty = abs(math.log2(max(aspect_ratio, 1e-6)))
    score += max(0.0, 24.0 - aspect_penalty * 8.0)
    if 300 <= candidate.width <= 5000 and 300 <= candidate.height <= 5000:
        score += 10.0
    if candidate.width >= 800 or candidate.height >= 800:
        score += 4.0
    if candidate.width == candidate.height:
        score += 2.0
    return score


def upload_url(filename: str, base_url: str) -> str:
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{base_url.rstrip('/')}/{digest[0]}/{digest[:2]}/{quote(filename, safe='')}"


def source_id(filename: str) -> str:
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()[:12]
    stem = filename.rsplit(".", 1)[0]
    slug = _slugify(stem) or "svg"
    return f"commons_dump_{digest}_{slug}"[:120]
