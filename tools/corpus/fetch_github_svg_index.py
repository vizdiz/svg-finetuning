"""Fetch GitHub SVG file candidates and write a JSONL index file.

With a GITHUB_TOKEN, uses the Code Search API (accurate, fast).
Without a token, falls back to the Git Tree API on a curated list of
SVG-rich public repositories (unauthenticated, 60 req/hr rate limit).

Writes a JSONL index consumable by:
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


_GH_API = "https://api.github.com"
_PAGE_SIZE = 100
_AUTH_DELAY_S = 1.0
_ANON_DELAY_S = 2.5  # ~60 req/hr unauthenticated
_MAX_SEARCH_RESULTS = 1000  # GitHub Search API hard cap


# Public repos known to contain many SVG diagrams / icons.
# Repo format: "owner/name"
_KNOWN_SVG_REPOS = [
    "kubernetes/website",
    "cncf/artwork",
    "cncf/landscape",
    "docker/docs",
    "grafana/grafana",
    "prometheus/prometheus",
    "istio/istio.io",
    "envoyproxy/envoy",
    "open-telemetry/opentelemetry.io",
    "hashicorp/terraform",
    "hashicorp/vault",
    "hashicorp/consul",
    "helm/helm",
    "argo-helm/argo-helm",
    "fluxcd/website",
    "crossplane/crossplane",
    "cert-manager/website",
    "linkerd/website",
    "dapr/dapr",
    "containerd/containerd",
    "etcd-io/website",
    "spiffe/spiffe.io",
    "thanos-io/thanos",
    "loki/loki",
    "jaegertracing/jaeger",
    "temporalio/temporal",
    "apache/airflow",
    "apache/arrow",
    "apache/beam",
    "apache/flink",
    "apache/kafka",
    "apache/spark",
    "tensorflow/tensorflow",
    "pytorch/pytorch",
    "huggingface/transformers",
    "microsoft/vscode",
    "microsoft/onnxruntime",
    "openai/openai-cookbook",
    "langchain-ai/langchain",
    "netdata/netdata",
    "vitessio/vitess",
    "cockroachdb/cockroach",
    "tikv/tikv",
    "pingcap/tidb",
    "vectordotdev/vector",
    "redpanda-data/redpanda",
    "nats-io/nats.docs",
    "rabbitmq/rabbitmq-server",
    "open-policy-agent/opa",
    "falcosecurity/falco",
    "aquasecurity/trivy",
    "cilium/cilium",
    "coredns/coredns",
    "metallb/metallb",
    "rancher/rancher",
    "k3s-io/k3s",
    "k0sproject/k0s",
    "rook/rook",
    "longhorn/longhorn",
    "minio/minio",
    "restic/restic",
    "borgbackup/borg",
    "oam-dev/spec",
    "score-spec/spec",
    "dagger/dagger",
    "earthly/earthly",
    "bazelbuild/bazel",
    "bufbuild/buf",
    "grpc/grpc",
    "protocolbuffers/protobuf",
    "graphql/graphql-spec",
    "opencontainers/image-spec",
    "opencontainers/runtime-spec",
    "sigstore/cosign",
    "buildpacks/spec",
    "knative/docs",
    "cloudevents/spec",
    "tektoncd/pipeline",
    "pipelineai/pipeline",
    "mlflow/mlflow",
    "kubeflow/website",
    "ray-project/ray",
    "bentoml/BentoML",
    "seldon-deploy/seldon-core",
    "feast-dev/feast",
    "zenml-io/zenml",
    "google/leveldb",
    "google/jax",
    "deepmind/graphcast",
    "facebookresearch/fairseq",
    "microsoft/DeepSpeed",
    "NVIDIA/apex",
    "triton-lang/triton",
]


def _headers(token: str | None) -> dict[str, str]:
    h: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _tree_to_rows(
    client: "httpx.Client",
    repo: str,
    token: str | None,
    delay: float,
) -> list[dict]:
    url = f"{_GH_API}/repos/{repo}/git/trees/HEAD"
    time.sleep(delay)
    resp = client.get(url, params={"recursive": "1"}, headers=_headers(token), timeout=30.0)
    if resp.status_code in (404, 451):
        return []
    resp.raise_for_status()
    data = resp.json()
    rows: list[dict] = []
    for item in data.get("tree", []):
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if not path.lower().endswith(".svg"):
            continue
        owner, name = repo.split("/", 1)
        raw_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{path}"
        rows.append({
            "raw_url": raw_url,
            "repo": repo,
            "path": path,
            "sha": item.get("sha", ""),
            "license": None,
            "repo_stars": None,
            "title": Path(path).name,
            "source": "github",
        })
    return rows


def _code_search_rows(
    client: "httpx.Client",
    query: str,
    token: str,
    limit: int,
    delay: float,
) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    page = 1
    while len(rows) < limit:
        time.sleep(delay)
        resp = client.get(
            f"{_GH_API}/search/code",
            params={"q": query, "per_page": _PAGE_SIZE, "page": page},
            headers=_headers(token),
            timeout=30.0,
        )
        if resp.status_code == 422:
            break
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        for item in items:
            sha = item.get("sha", "")
            if sha in seen:
                continue
            seen.add(sha)
            repo_obj = item.get("repository", {})
            rows.append({
                "raw_url": item.get("download_url") or item.get("html_url", ""),
                "repo": repo_obj.get("full_name", ""),
                "path": item.get("path", ""),
                "sha": sha,
                "license": (repo_obj.get("license") or {}).get("spdx_id"),
                "repo_stars": repo_obj.get("stargazers_count", 0),
                "title": item.get("name", ""),
                "source": "github",
            })
        total = data.get("total_count", 0)
        print(f"  Got {len(items)} items (total={total}, collected={len(rows)})")
        if page * _PAGE_SIZE >= min(total, _MAX_SEARCH_RESULTS):
            break
        page += 1
    return rows


def fetch_github_svg_index(*, limit: int, token: str | None) -> list[dict]:
    import httpx

    delay = _AUTH_DELAY_S if token else _ANON_DELAY_S
    rows: list[dict] = []
    seen_sha: set[str] = set()

    with httpx.Client() as client:
        if token:
            queries = [
                "extension:svg language:SVG pushed:>2023-01-01",
                "filename:diagram.svg",
                "filename:architecture.svg",
                "filename:schema.svg",
            ]
            for q in queries:
                if len(rows) >= limit:
                    break
                print(f"  GitHub code search: {q!r}", flush=True)
                for r in _code_search_rows(client, q, token, limit - len(rows), delay):
                    key = r["sha"] or r["raw_url"]
                    if key not in seen_sha:
                        seen_sha.add(key)
                        rows.append(r)
        else:
            print("No GITHUB_TOKEN — using tree API on curated SVG repos", flush=True)
            for repo in _KNOWN_SVG_REPOS:
                if len(rows) >= limit:
                    break
                print(f"  Tree: {repo}", flush=True)
                repo_rows = _tree_to_rows(client, repo, token, delay)
                for r in repo_rows:
                    key = r["sha"] or r["raw_url"]
                    if key not in seen_sha:
                        seen_sha.add(key)
                        rows.append(r)
                print(f"  Got {len(repo_rows)} SVGs from {repo} (total={len(rows)})")

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
