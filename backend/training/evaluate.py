"""
training.evaluate

Deterministic dataset and output evaluation for SVG/diagram-IR training runs.

The pre-training checks answer whether a manifest is trainable: prompts are
present, targets parse, SVGs pass safety validation, and IR targets pass the
structural quality gate. Optional prediction files can use the same validators
after inference is wired into the training workflow.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from dataclasses import asdict, dataclass, field
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

from lxml import etree

from backend.dataset_pipeline.processing.validator import validate_svg, validate_svg_detailed
from backend.training.dataset_interface import DatasetManifest, TrainingRecord, read_manifest_from_s3
from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import DiagramIRDocument
from backend.training.diagram_quality import assess_compiled_svg, assess_diagram_ir


@dataclass
class EvaluationReport:
    model_artifacts: str
    val_records: int
    svg_valid_rate: float = 0.0
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticResult:
    id: str
    quality_score: float
    validity: float
    geometry: float
    semantic_alignment: float | None
    complexity: float
    editability: float
    flaws: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticDiagnostic:
    id: str
    semantic_alignment: float
    layout_fidelity: float
    label_fidelity: float
    connection_fidelity: float
    geometry_fidelity: float
    diagram_usefulness: float
    missing_requirements: list[str] = field(default_factory=list)
    hallucinated_elements: list[str] = field(default_factory=list)
    flaws: list[str] = field(default_factory=list)
    notes: str = ""


_SEMANTIC_MODEL = "claude-sonnet-4-20250514"
_SEMANTIC_SYSTEM_PROMPT = """You are evaluating whether a rendered SVG diagram matches its text prompt.
Score only what is visible in the image. Do not infer intent that is not visible.
Return strict JSON with numeric scores from 0.0 to 1.0 and short string arrays."""
_SEMANTIC_USER_PROMPT = """Prompt:
{prompt}

Evaluate the rendered SVG against the prompt. Focus on technical-diagram fidelity:
- semantic_alignment: overall prompt/image match
- layout_fidelity: relative placement, grouping, containment, scale, alignment
- label_fidelity: expected text labels are present, readable, and attached to the right elements
- connection_fidelity: arrows/edges/routing/direction/edge labels match the prompt
- geometry_fidelity: shapes are coherent, non-degenerate, and visually usable
- diagram_usefulness: a person could reconstruct or use the intended diagram from this output

Return only this JSON object:
{{
  "semantic_alignment": 0.0,
  "layout_fidelity": 0.0,
  "label_fidelity": 0.0,
  "connection_fidelity": 0.0,
  "geometry_fidelity": 0.0,
  "diagram_usefulness": 0.0,
  "missing_requirements": [],
  "hallucinated_elements": [],
  "flaws": [],
  "notes": ""
}}"""


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def _iter_manifest_records(manifest: DatasetManifest, manifest_path: Path | None = None) -> Iterable[TrainingRecord]:
    for uri in manifest.files:
        if uri.startswith("s3://"):
            bucket, key = uri[len("s3://") :].split("/", 1)
            import boto3

            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=bucket, Key=key)
            for line in response["Body"].iter_lines():
                if line:
                    yield TrainingRecord.from_dict(json.loads(line))
            continue

        path = Path(uri)
        if not path.is_absolute() and manifest_path is not None:
            path = manifest_path.parent / path
        for payload in _read_jsonl(path):
            yield TrainingRecord.from_dict(payload)


def _ir_document(diagram_ir: dict[str, Any]) -> DiagramIRDocument | None:
    try:
        return DiagramIRDocument.from_dict(diagram_ir)
    except Exception:
        return None


def _record_svg(record: TrainingRecord) -> str:
    if record.svg:
        return record.svg
    if record.diagram_ir is None:
        raise ValueError("record has neither SVG nor diagram_ir")
    document = DiagramIRDocument.from_dict(record.diagram_ir)
    return compile_diagram_ir(document)


def _render_svg_png_base64(svg: str) -> str:
    import cairosvg

    png_buffer = io.BytesIO()
    cairosvg.svg2png(bytestring=svg.encode(), write_to=png_buffer, output_width=800)
    return base64.b64encode(png_buffer.getvalue()).decode("ascii")


def _response_text(response: object) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", text)
        if match is None:
            raise ValueError("semantic response did not contain a JSON object")
        text = match.group(0)
    return json.loads(text)


def _semantic_diagnostic_from_payload(record_id: str, payload: dict[str, Any]) -> SemanticDiagnostic:
    def score(key: str) -> float:
        try:
            return _clamp(float(payload.get(key, 0.0)))
        except (TypeError, ValueError):
            return 0.0

    def strings(key: str) -> list[str]:
        values = payload.get(key, [])
        if not isinstance(values, list):
            return []
        return [str(value) for value in values if str(value).strip()]

    return SemanticDiagnostic(
        id=record_id,
        semantic_alignment=score("semantic_alignment"),
        layout_fidelity=score("layout_fidelity"),
        label_fidelity=score("label_fidelity"),
        connection_fidelity=score("connection_fidelity"),
        geometry_fidelity=score("geometry_fidelity"),
        diagram_usefulness=score("diagram_usefulness"),
        missing_requirements=strings("missing_requirements"),
        hallucinated_elements=strings("hallucinated_elements"),
        flaws=strings("flaws"),
        notes=str(payload.get("notes", "")),
    )


def score_semantic_alignment(
    record: TrainingRecord,
    client: object | None = None,
    model: str = _SEMANTIC_MODEL,
) -> SemanticDiagnostic:
    if not record.prompt.strip():
        return SemanticDiagnostic(
            id=record.id,
            semantic_alignment=0.0,
            layout_fidelity=0.0,
            label_fidelity=0.0,
            connection_fidelity=0.0,
            geometry_fidelity=0.0,
            diagram_usefulness=0.0,
            flaws=["missing_prompt"],
            notes="Semantic alignment requires a prompt.",
        )

    try:
        image_data = _render_svg_png_base64(_record_svg(record))
    except Exception as exc:
        return SemanticDiagnostic(
            id=record.id,
            semantic_alignment=0.0,
            layout_fidelity=0.0,
            label_fidelity=0.0,
            connection_fidelity=0.0,
            geometry_fidelity=0.0,
            diagram_usefulness=0.0,
            flaws=[f"render_failed: {type(exc).__name__}: {exc}"],
            notes="Could not render SVG for semantic evaluation.",
        )

    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        system=_SEMANTIC_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": _SEMANTIC_USER_PROMPT.format(prompt=record.prompt),
                    },
                ],
            }
        ],
    )
    return _semantic_diagnostic_from_payload(record.id, _parse_json_object(_response_text(response)))


def evaluate_semantic_alignment(
    records: Iterable[TrainingRecord],
    output_path: Path,
    sample_size: int,
    client: object | None = None,
    model: str = _SEMANTIC_MODEL,
) -> dict[str, Any]:
    if client is None:
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            timeout=120.0,
            max_retries=2,
        )

    diagnostics: list[SemanticDiagnostic] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for index, record in enumerate(records):
            if index >= sample_size:
                break
            try:
                diagnostic = score_semantic_alignment(record, client=client, model=model)
            except Exception as exc:
                diagnostic = SemanticDiagnostic(
                    id=record.id,
                    semantic_alignment=0.0,
                    layout_fidelity=0.0,
                    label_fidelity=0.0,
                    connection_fidelity=0.0,
                    geometry_fidelity=0.0,
                    diagram_usefulness=0.0,
                    flaws=[f"semantic_eval_failed: {type(exc).__name__}: {exc}"],
                    notes="Semantic evaluator failed for this record.",
                )
            diagnostics.append(diagnostic)
            handle.write(json.dumps(asdict(diagnostic), sort_keys=True) + "\n")

    total = len(diagnostics)
    flaw_counts: Counter[str] = Counter()
    for diagnostic in diagnostics:
        flaw_counts.update(diagnostic.flaws)

    return {
        "sample_size": total,
        "semantic_alignment": _rate(sum(item.semantic_alignment for item in diagnostics), total),
        "layout_fidelity": _rate(sum(item.layout_fidelity for item in diagnostics), total),
        "label_fidelity": _rate(sum(item.label_fidelity for item in diagnostics), total),
        "connection_fidelity": _rate(sum(item.connection_fidelity for item in diagnostics), total),
        "geometry_fidelity": _rate(sum(item.geometry_fidelity for item in diagnostics), total),
        "diagram_usefulness": _rate(sum(item.diagram_usefulness for item in diagnostics), total),
        "flaw_counts": dict(flaw_counts.most_common()),
    }


def _local_name(tag: str) -> str:
    try:
        return etree.QName(tag).localname.lower()
    except ValueError:
        return ""


def _shape_counts(svg: str) -> dict[str, int]:
    try:
        root = etree.fromstring(svg.encode(), etree.XMLParser(resolve_entities=False, no_network=True))
    except etree.XMLSyntaxError:
        return {}

    counts: Counter[str] = Counter()
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag:
            counts[tag] += 1
    primitive_tags = ("rect", "circle", "ellipse", "line", "polyline", "polygon", "path", "text")
    counts["primitive_elements"] = sum(counts[tag] for tag in primitive_tags)
    counts["shape_elements"] = sum(counts[tag] for tag in primitive_tags if tag != "text")
    return dict(counts)


def _svg_complexity_score(stats: dict[str, Any], shape_counts: dict[str, int]) -> float:
    nodes = int(stats.get("nodes", 0) or 0)
    paths = int(stats.get("paths", 0) or 0)
    bytes_len = int(stats.get("bytes", 0) or 0)
    score = 1.0
    if nodes < 6:
        score -= 0.35
    if bytes_len < 500:
        score -= 0.25
    if nodes > 2000:
        score -= min(0.35, (nodes - 2000) / 8000)
    if paths > 800:
        score -= min(0.25, (paths - 800) / 4000)
    if int(shape_counts.get("primitive_elements", 0) or 0) == 0:
        score -= 0.25
    return _clamp(score)


def _svg_geometry_score(valid: bool, stats: dict[str, Any]) -> float:
    score = 1.0 if valid else 0.25
    aspect_ratio = stats.get("aspect_ratio")
    if not stats.get("has_viewbox") and not stats.get("has_width_height"):
        score -= 0.35
    if isinstance(aspect_ratio, (int, float)) and (aspect_ratio < 0.1 or aspect_ratio > 10.0):
        score -= 0.25
    if int(stats.get("hidden_elements_or_attrs", 0) or 0) > 100:
        score -= 0.15
    return _clamp(score)


def _svg_editability_score(stats: dict[str, Any], shape_counts: dict[str, int]) -> float:
    primitives = int(shape_counts.get("primitive_elements", 0) or 0)
    paths = int(stats.get("paths", 0) or 0)
    nodes = int(stats.get("nodes", 0) or 0)
    groups = int(shape_counts.get("g", 0) or 0)
    score = 0.65
    if primitives:
        score += 0.15
    if groups:
        score += 0.1
    if paths and primitives and paths / primitives > 0.9 and primitives > 20:
        score -= 0.2
    if nodes > 2000:
        score -= 0.2
    return _clamp(score)


def _ir_scores(document: DiagramIRDocument, quality_accepted: bool) -> tuple[float, float, dict[str, Any]]:
    node_count = len(document.nodes)
    edge_count = len(document.edges)
    group_count = len(document.groups)
    geometry = 1.0 if quality_accepted else 0.45
    complexity = 0.75
    if node_count >= 2:
        complexity += 0.15
    if edge_count:
        complexity += 0.1
    if node_count > 80 or edge_count > 120:
        complexity -= 0.25
    editability = 0.7
    if group_count:
        editability += 0.1
    if all(node.id and node.kind for node in document.nodes):
        editability += 0.1
    if all(edge.source and edge.target for edge in document.edges):
        editability += 0.1
    return (
        _clamp(geometry),
        _clamp(complexity),
        {
            "ir_nodes": node_count,
            "ir_edges": edge_count,
            "ir_groups": group_count,
            "ir_editability": _clamp(editability),
        },
    )


def diagnose_record(record: TrainingRecord) -> DiagnosticResult:
    flaws: set[str] = set()
    stats: dict[str, Any] = {}
    valid_targets = 0
    geometry_scores: list[float] = []
    complexity_scores: list[float] = []
    editability_scores: list[float] = []

    if not record.prompt.strip():
        flaws.add("missing_prompt")

    if record.svg:
        svg_result = validate_svg_detailed(record.svg)
        stats.update({f"svg_{key}": value for key, value in svg_result.stats.items()})
        shape_counts = _shape_counts(record.svg)
        stats.update({f"svg_{key}_count": value for key, value in shape_counts.items()})
        if svg_result.ok:
            valid_targets += 1
            quality = assess_compiled_svg(record.svg)
            if not quality.accepted:
                flaws.update(f"svg_{reason}" for reason in quality.reasons)
        else:
            flaws.update(f"svg_{reason}" for reason in svg_result.hard_reject_reasons)
        flaws.update(f"svg_warning_{warning.split(':', 1)[0]}" for warning in svg_result.warnings)
        geometry_scores.append(_svg_geometry_score(svg_result.ok, svg_result.stats))
        complexity_scores.append(_svg_complexity_score(svg_result.stats, shape_counts))
        editability_scores.append(_svg_editability_score(svg_result.stats, shape_counts))

    if record.diagram_ir is not None:
        document = _ir_document(record.diagram_ir)
        if document is None:
            flaws.add("ir_schema_invalid")
            geometry_scores.append(0.2)
            complexity_scores.append(0.2)
            editability_scores.append(0.2)
        else:
            valid_targets += 1
            quality = assess_diagram_ir(document)
            if not quality.accepted:
                flaws.update(f"ir_{reason}" for reason in quality.reasons)
            geometry, complexity, ir_stats = _ir_scores(document, quality.accepted)
            stats.update(ir_stats)
            geometry_scores.append(geometry)
            complexity_scores.append(complexity)
            editability_scores.append(ir_stats["ir_editability"])

    if not record.svg and record.diagram_ir is None:
        flaws.add("missing_target")

    validity = 1.0 if valid_targets else 0.0
    geometry = sum(geometry_scores) / len(geometry_scores) if geometry_scores else 0.0
    complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.0
    editability = sum(editability_scores) / len(editability_scores) if editability_scores else 0.0
    quality_score = sum((validity, geometry, complexity, editability)) / 4

    return DiagnosticResult(
        id=record.id,
        quality_score=_clamp(quality_score),
        validity=validity,
        geometry=_clamp(geometry),
        semantic_alignment=None,
        complexity=_clamp(complexity),
        editability=_clamp(editability),
        flaws=sorted(flaws),
        stats=stats,
    )


def _diagnostic_summary(diagnostics: list[DiagnosticResult]) -> dict[str, Any]:
    flaw_counts: Counter[str] = Counter()
    for diagnostic in diagnostics:
        flaw_counts.update(diagnostic.flaws)
    total = len(diagnostics)
    return {
        "quality_score": _rate(sum(item.quality_score for item in diagnostics), total),
        "validity": _rate(sum(item.validity for item in diagnostics), total),
        "geometry": _rate(sum(item.geometry for item in diagnostics), total),
        "semantic_alignment": None,
        "complexity": _rate(sum(item.complexity for item in diagnostics), total),
        "editability": _rate(sum(item.editability for item in diagnostics), total),
        "flaw_counts": dict(flaw_counts.most_common()),
        "flaw_free_rate": _rate(sum(1 for item in diagnostics if not item.flaws), total),
    }


def evaluate_records(
    records: Iterable[TrainingRecord],
    diagnostic_output_path: Path | None = None,
) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    prompt_words = 0
    prompt_chars = 0
    svg_reasons: Counter[str] = Counter()
    ir_reasons: Counter[str] = Counter()
    node_counts: list[int] = []
    edge_counts: list[int] = []
    diagnostics: list[DiagnosticResult] = []

    diagnostic_handle = diagnostic_output_path.open("w") if diagnostic_output_path is not None else None
    try:
        for record in records:
            diagnostic = diagnose_record(record)
            diagnostics.append(diagnostic)
            if diagnostic_handle is not None:
                diagnostic_handle.write(json.dumps(asdict(diagnostic), sort_keys=True) + "\n")
            counter["record_count"] += 1
            if record.prompt.strip():
                counter["prompt_present"] += 1
                prompt_words += len(record.prompt.split())
                prompt_chars += len(record.prompt)
            if record.source:
                sources[record.source] += 1

            svg_valid = False
            if record.svg:
                counter["target_svg_present"] += 1
                svg_valid, reason = validate_svg(record.svg)
                if svg_valid:
                    counter["target_svg_valid"] += 1
                    quality = assess_compiled_svg(record.svg)
                    if quality.accepted:
                        counter["target_svg_quality_accepted"] += 1
                else:
                    svg_reasons[reason] += 1

            ir_valid = False
            if record.diagram_ir is not None:
                counter["target_ir_present"] += 1
                document = _ir_document(record.diagram_ir)
                if document is None:
                    ir_reasons["schema_invalid"] += 1
                else:
                    ir_valid = True
                    counter["target_ir_valid"] += 1
                    quality = assess_diagram_ir(document)
                    if quality.accepted:
                        counter["target_ir_quality_accepted"] += 1
                    else:
                        ir_reasons.update(quality.reasons or ["quality_rejected"])
                    node_counts.append(int(quality.metrics.get("node_count", len(document.nodes))))
                    edge_counts.append(int(quality.metrics.get("edge_count", len(document.edges))))

            if svg_valid or ir_valid:
                counter["target_any_valid"] += 1
    finally:
        if diagnostic_handle is not None:
            diagnostic_handle.close()

    total = counter["record_count"]
    ir_present = counter["target_ir_present"]
    svg_present = counter["target_svg_present"]
    return {
        "record_count": total,
        "prompt_present": counter["prompt_present"],
        "prompt_present_rate": _rate(counter["prompt_present"], total),
        "avg_prompt_words": _rate(prompt_words, total),
        "avg_prompt_chars": _rate(prompt_chars, total),
        "source_counts": dict(sorted(sources.items())),
        "target_svg_present": svg_present,
        "target_svg_valid": counter["target_svg_valid"],
        "target_svg_valid_rate": _rate(counter["target_svg_valid"], svg_present),
        "target_svg_quality_accepted": counter["target_svg_quality_accepted"],
        "target_svg_quality_accept_rate": _rate(counter["target_svg_quality_accepted"], svg_present),
        "target_svg_reject_reasons": dict(svg_reasons.most_common()),
        "target_ir_present": ir_present,
        "target_ir_valid": counter["target_ir_valid"],
        "target_ir_valid_rate": _rate(counter["target_ir_valid"], ir_present),
        "target_ir_quality_accepted": counter["target_ir_quality_accepted"],
        "target_ir_quality_accept_rate": _rate(counter["target_ir_quality_accepted"], ir_present),
        "target_ir_reject_reasons": dict(ir_reasons.most_common()),
        "target_any_valid": counter["target_any_valid"],
        "target_any_valid_rate": _rate(counter["target_any_valid"], total),
        "avg_ir_nodes": _rate(sum(node_counts), len(node_counts)),
        "avg_ir_edges": _rate(sum(edge_counts), len(edge_counts)),
        "diagnostics": _diagnostic_summary(diagnostics),
    }


def _prediction_records(path: Path) -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    for payload in _read_jsonl(path):
        records.append(
            TrainingRecord.from_dict(
                {
                    "prompt": payload.get("prompt", ""),
                    "svg": payload.get("svg") or payload.get("prediction") or payload.get("completion") or "",
                    "diagram_ir": payload.get("diagram_ir"),
                    "id": payload.get("id", ""),
                    "source": payload.get("source", "prediction"),
                }
            )
        )
    return records


def evaluate_predictions(path: Path, diagnostic_output_path: Path | None = None) -> dict[str, Any]:
    records = _prediction_records(path)
    return evaluate_records(records, diagnostic_output_path=diagnostic_output_path)


def build_report(
    model_artifacts: str,
    val_records: int,
    dataset_metrics: dict[str, Any] | None = None,
    prediction_metrics: dict[str, Any] | None = None,
) -> EvaluationReport:
    metrics: dict[str, Any] = {}
    notes: list[str] = []
    if dataset_metrics is not None:
        metrics["dataset"] = dataset_metrics
        val_records = int(dataset_metrics.get("record_count", val_records))
    else:
        notes.append("No manifest supplied; only legacy val-count metric is available")
    if prediction_metrics is not None:
        metrics["predictions"] = prediction_metrics

    svg_valid_rate = 0.0
    if dataset_metrics is not None:
        svg_valid_rate = float(dataset_metrics.get("target_svg_valid_rate", 0.0))
    elif val_records:
        svg_valid_rate = 1.0

    return EvaluationReport(
        model_artifacts=model_artifacts,
        val_records=val_records,
        svg_valid_rate=svg_valid_rate,
        notes=notes,
        metrics=metrics,
    )


def _load_manifest(path: str | None, data_bucket: str | None) -> tuple[DatasetManifest | None, Path | None]:
    if data_bucket:
        return read_manifest_from_s3(data_bucket), None
    if path is None:
        return None, None
    manifest_path = Path(path)
    return DatasetManifest.from_json(manifest_path.read_text()), manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an SVG/diagram-IR training run")
    parser.add_argument("--model-artifacts", required=True, help="Path or S3 URI to model artifacts")
    parser.add_argument("--val-count", type=int, default=0, help="Fallback validation record count")
    parser.add_argument("--manifest", help="Local dataset_manifest.json to evaluate")
    parser.add_argument("--data-bucket", help="Read train/dataset_manifest.json from this S3 bucket")
    parser.add_argument("--predictions", help="Optional JSONL file of model outputs to evaluate")
    parser.add_argument("--semantic-sample-size", type=int, default=0, help="Number of records to VLM-score")
    parser.add_argument("--semantic-model", default=_SEMANTIC_MODEL, help="Anthropic model for semantic scoring")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation outputs")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, manifest_path = _load_manifest(args.manifest, args.data_bucket)
    dataset_metrics = None
    dataset_records: list[TrainingRecord] | None = None
    if manifest is not None:
        dataset_records = list(_iter_manifest_records(manifest, manifest_path))
        dataset_metrics = evaluate_records(
            dataset_records,
            diagnostic_output_path=output_dir / "dataset_diagnostics.jsonl",
        )
        if args.semantic_sample_size > 0:
            dataset_metrics["semantic"] = evaluate_semantic_alignment(
                dataset_records,
                output_path=output_dir / "semantic_diagnostics.jsonl",
                sample_size=args.semantic_sample_size,
                model=args.semantic_model,
            )

    prediction_metrics = None
    if args.predictions:
        prediction_records = _prediction_records(Path(args.predictions))
        prediction_metrics = evaluate_records(
            prediction_records,
            diagnostic_output_path=output_dir / "prediction_diagnostics.jsonl",
        )
        if args.semantic_sample_size > 0:
            prediction_metrics["semantic"] = evaluate_semantic_alignment(
                prediction_records,
                output_path=output_dir / "prediction_semantic_diagnostics.jsonl",
                sample_size=args.semantic_sample_size,
                model=args.semantic_model,
            )
    report = build_report(args.model_artifacts, args.val_count, dataset_metrics, prediction_metrics)

    output_path = output_dir / "evaluation.json"
    output_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
