"""Fetch GitHub SVG file candidates from the GitHub Code Search API.

Writes a JSONL index file consumable by:
    svg-corpus-build-bulk-candidates --source github --index-jsonl ...

Usage:
    svg-corpus-fetch-github-index \
        --output pipeline_output/corpus/github_svg_index.jsonl \
        --limit 7500
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


_GH_SEARCH_URL = "https://api.github.com/search/code"
_PAGE_SIZE = 100
_ANON_DELAY_S = 12.0
_AUTH_DELAY_S = 2.0
_MAX_RESULTS = 1000  # GitHub Search API hard cap per query


def _search_page(
    client: "httpx.Client",
    query: str,
    page: int,
    per_page: int,
) -> tuple[list[dict], int]:
    resp = client.get(
        _GH_SEARCH_URL,
        params={"q": query, "per_page": per_page, "page": page},
        headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
        timeout=30.0,
    )
    if resp.status_code == 422:
        return [], 0
    resp.raise_for_status()
    data = resp.json()
    return data.get("items", []), data.get("total_count", 0)


def _item_to_row(item: dict) -> dict:
    repo = item.get("repository", {})
    return {
        "raw_url": item.get("download_url") or item.get("html_url", ""),
        "repo": repo.get("full_name", ""),
        "path": item.get("path", ""),
        "sha": item.get("sha", ""),
        "license": repo.get("license", {}).get("spdx_id") if repo.get("license") else None,
        "repo_stars": repo.get("stargazers_count", 0),
        "repo_language": repo.get("language", ""),
        "title": item.get("name", ""),
        "source": "github",
    }


def fetch_github_svg_index(*, limit: int, token: str | None) -> list[dict]:
    import httpx

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    delay = _AUTH_DELAY_S if token else _ANON_DELAY_S

    # Use multiple targeted queries to beat the 1000-result cap
    queries = [
        "extension:svg language:SVG pushed:>2023-01-01",
        "filename:diagram.svg",
        "filename:architecture.svg",
        "filename:flowchart.svg",
        "filename:schema.svg",
        "filename:network.svg",
    ]

    rows: list[dict] = []
    seen: set[str] = set()

    with httpx.Client(headers=headers) as client:
        for query in queries:
            if len(rows) >= limit:
                break
            page = 1
            while len(rows) < limit:
                print(f"  GitHub query={query!r} page={page} ...", flush=True)
                items, total = _search_page(client, query, page, _PAGE_SIZE)
                if not items:
                    break
                for item in items:
                    row = _item_to_row(item)
                    key = row["sha"] or row["raw_url"]
                    if key and key not in seen:
                        seen.add(key)
                        rows.append(row)
                print(f"  Got {len(items)} items (total={total}, collected={len(rows)})")
                if page * _PAGE_SIZE >= min(total, _MAX_RESULTS) or not items:
                    break
                page += 1
                time.sleep(delay)

    return rows[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=7500)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    args = parser.parse_args()

    rows = fetch_github_svg_index(limit=args.limit, token=args.token or None)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"Wrote {len(rows)} GitHub SVG rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
