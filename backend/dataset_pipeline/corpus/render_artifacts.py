from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

from PIL import Image
import cairosvg

from backend.dataset_pipeline.corpus.schema import utc_now, write_jsonl


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def _average_hash(image_path: Path, hash_size: int = 8) -> str:
    with Image.open(image_path) as image:
        gray = image.convert("L").resize((hash_size, hash_size))
        pixels = list(gray.tobytes())
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= mean else "0" for pixel in pixels)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def build_render_artifacts(
    *,
    records_path: Path,
    output_dir: Path,
    max_records: int | None = None,
    duplicate_hamming_threshold: int = 4,
    thumbnail_width: int = 512,
) -> dict[str, Any]:
    rows = _read_jsonl(records_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir = output_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    seen_hashes: dict[str, str] = {}
    near_duplicates = 0
    rendered = 0

    for index, row in enumerate(rows):
        if max_records is not None and index >= max_records:
            break
        svg = str(row.get("svg") or "")
        if not svg:
            continue
        record_id = str(row.get("id") or f"record-{index}")
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in record_id)[:120]
        png_path = thumbnails_dir / f"{safe_id}.png"
        try:
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), output_width=thumbnail_width)
            png_bytes = png_path.read_bytes()
            render_sha256 = hashlib.sha256(png_bytes).hexdigest()
            render_ahash = _average_hash(png_path)
            duplicate_of = ""
            for prior_hash, prior_id in seen_hashes.items():
                if _hamming(render_ahash, prior_hash) <= duplicate_hamming_threshold:
                    duplicate_of = prior_id
                    near_duplicates += 1
                    break
            if not duplicate_of:
                seen_hashes[render_ahash] = record_id
            artifacts.append(
                {
                    "id": record_id,
                    "source": row.get("source", ""),
                    "thumbnail_path": str(png_path),
                    "render_sha256": render_sha256,
                    "render_ahash": render_ahash,
                    "near_duplicate_of": duplicate_of,
                    "render_ok": True,
                }
            )
            rendered += 1
        except Exception as exc:
            failures.append({"id": record_id, "reason": f"{type(exc).__name__}: {exc}"})
            artifacts.append({"id": record_id, "source": row.get("source", ""), "render_ok": False})

    artifacts_path = output_dir / "render_artifacts.jsonl"
    write_jsonl(artifacts_path, artifacts)
    summary = {
        "created_at": utc_now(),
        "records_path": str(records_path),
        "artifact_count": len(artifacts),
        "rendered": rendered,
        "failed": len(failures),
        "near_duplicates": near_duplicates,
        "duplicate_hamming_threshold": duplicate_hamming_threshold,
        "thumbnails_dir": str(thumbnails_dir),
        "artifacts": str(artifacts_path),
        "sample_failures": failures[:50],
    }
    summary_path = output_dir / "render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
