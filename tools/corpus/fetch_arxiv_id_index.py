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


def _fetch_page(query: str, start: int, max_results: int) -> tuple[list[str], int]:
    import httpx
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = httpx.get(_API_BASE, params=params, timeout=60.0)
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


def fetch_arxiv_ids(
    *,
    categories: list[str],
    limit: int,
    start_date: str | None,
    end_date: str | None,
) -> list[str]:
    date_filter = ""
    if start_date or end_date:
        lo = (start_date or "19900101").replace("-", "")
        hi = (end_date or "20991231").replace("-", "")
        date_filter = f" AND submittedDate:[{lo}0000 TO {hi}2359]"
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    query = f"({cat_query}){date_filter}"
    print(f"Query: {query}")
    all_ids: list[str] = []
    seen: set[str] = set()
    start = 0
    while len(all_ids) < limit:
        batch_size = min(_PAGE_SIZE, limit - len(all_ids))
        print(f"  Fetching start={start} batch={batch_size} ...", flush=True)
        ids, total = _fetch_page(query, start, batch_size)
        if not ids:
            break
        for i in ids:
            if i not in seen:
                seen.add(i)
                all_ids.append(i)
        print(f"  Got {len(ids)} IDs (total available: {total}, collected: {len(all_ids)})")
        start += len(ids)
        if start >= total:
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

    ids = fetch_arxiv_ids(
        categories=args.categories,
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids) + "\n")
    print(f"Wrote {len(ids)} arXiv IDs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
