"""Fetch arXiv paper IDs from the arXiv Atom API and write an ID list file.

Usage:
    svg-corpus-fetch-arxiv-index \
        --output pipeline_output/corpus/arxiv_ids.txt \
        --categories cs.CV cs.LG cs.AI cs.AR cs.CG math.CO \
        --limit 20000 \
        --start-date 2022-01-01 \
        --end-date 2025-12-31
"""
from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path


_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"
_OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
_API_BASE = "https://export.arxiv.org/api/query"
_PAGE_SIZE = 2000
_PAGE_DELAY_S = 4.0


_MAX_SAFE_OFFSET = 9000  # arXiv API returns 500 above ~10K


def _fetch_page(query: str, start: int, max_results: int, *, max_retries: int = 3) -> tuple[list[str], int]:
    import httpx

    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    for attempt in range(max_retries):
        try:
            resp = httpx.get(_API_BASE, params=params, timeout=60.0)
            if resp.status_code >= 500:
                if attempt < max_retries - 1:
                    print(f"  arXiv {resp.status_code} at start={start}, retry {attempt+1}/{max_retries}")
                    time.sleep(10.0 * (attempt + 1))
                    continue
                return [], 0
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            total_str = root.findtext(f"{{{_OPENSEARCH_NS}}}totalResults") or "0"
            total = int(total_str)
            ids: list[str] = []
            for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
                id_elem = entry.find(f"{{{_ATOM_NS}}}id")
                if id_elem is not None and id_elem.text:
                    raw = id_elem.text.strip().rstrip("/")
                    paper_id = raw.rsplit("/", 1)[-1]
                    ids.append(paper_id.split("v")[0])
            return ids, total
        except Exception as exc:
            if attempt < max_retries - 1:
                print(f"  arXiv error at start={start}: {exc}, retry {attempt+1}")
                time.sleep(10.0)
            else:
                return [], 0
    return [], 0


def _year_month_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split date range into monthly chunks to avoid arXiv high-offset 500s."""
    from datetime import date, timedelta

    lo = date.fromisoformat(start_date)
    hi = date.fromisoformat(end_date)
    ranges: list[tuple[str, str]] = []
    cur = lo.replace(day=1)
    while cur <= hi:
        # end of month
        if cur.month == 12:
            next_month = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            next_month = cur.replace(month=cur.month + 1, day=1)
        end = min(next_month - timedelta(days=1), hi)
        ranges.append((cur.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        cur = next_month
    return ranges


def fetch_arxiv_ids(
    *,
    categories: list[str],
    limit: int,
    start_date: str | None,
    end_date: str | None,
    checkpoint_path: "Path | None" = None,
) -> list[str]:
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    all_ids: list[str] = []
    seen: set[str] = set()

    # Load checkpoint if available
    if checkpoint_path and checkpoint_path.exists():
        existing = checkpoint_path.read_text().strip().splitlines()
        for i in existing:
            if i and i not in seen:
                seen.add(i)
                all_ids.append(i)
        print(f"Resumed from checkpoint: {len(all_ids)} IDs")

    if start_date and end_date:
        # Query month-by-month to avoid high-offset 500 errors
        date_ranges = _year_month_ranges(start_date, end_date)
    else:
        date_ranges = [(start_date or "", end_date or "")]

    for range_start, range_end in date_ranges:
        if len(all_ids) >= limit:
            break
        if range_start and range_end:
            lo = range_start.replace("-", "")
            hi = range_end.replace("-", "")
            date_filter = f" AND submittedDate:[{lo}0000 TO {hi}2359]"
        else:
            date_filter = ""
        query = f"({cat_query}){date_filter}"
        print(f"Query: {query[:100]}", flush=True)

        start = 0
        while len(all_ids) < limit and start < _MAX_SAFE_OFFSET:
            batch_size = min(_PAGE_SIZE, limit - len(all_ids))
            print(f"  Fetching start={start} batch={batch_size} ...", flush=True)
            ids, total = _fetch_page(query, start, batch_size)
            if not ids:
                break
            added = 0
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    all_ids.append(i)
                    added += 1
            print(f"  Got {len(ids)} IDs (+{added} new, total available: {total}, collected: {len(all_ids)})")

            # Checkpoint after each page
            if checkpoint_path:
                checkpoint_path.write_text("\n".join(all_ids) + "\n")

            start += len(ids)
            if start >= min(total, _MAX_SAFE_OFFSET):
                break
            time.sleep(_PAGE_DELAY_S)

    return all_ids[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--categories", nargs="+", default=["cs.CV", "cs.LG", "cs.AI", "cs.AR", "cs.CG", "math.CO"])
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Use output file as checkpoint so reruns skip already-fetched IDs
    ids = fetch_arxiv_ids(
        categories=args.categories,
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
        checkpoint_path=out,
    )
    out.write_text("\n".join(ids) + "\n")
    print(f"Wrote {len(ids)} arXiv IDs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
