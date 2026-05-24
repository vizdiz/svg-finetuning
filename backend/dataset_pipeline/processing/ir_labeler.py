"""
dataset_pipeline.processing.ir_labeler

Bridge from scraped SVGs to the diagram IR training labels.

The parser is intentionally narrow: it only accepts the compiler's SVG
dialect. That is useful for the IR-labeled path because it gives us a
deterministic structural gate and stable labels.

Before parsing, the SVG is passed through the safe normalizer so inherited
styles, no-op transforms, and formatting noise do not affect extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.dataset_pipeline.processing.types import RawSVG
from backend.dataset_pipeline.processing.svg_ir_extractor import (
    extract_primitives,
    infer_ir,
    normalize_svg,
    render_ir,
    roundtrip_ok,
    score_ir,
)


@dataclass(slots=True)
class IRLabelResult:
    accepted: bool
    reason: str


def label_raw_svg_ir(raw_svg: RawSVG) -> IRLabelResult:
    """
    Try to parse a raw SVG into the narrow diagram IR and attach the
    label payload to raw_svg.metadata when it succeeds.

    The label payload is kept small and JSON-serializable so downstream
    stages can persist it in manifests without re-parsing the SVG.
    """
    canonical_dom = normalize_svg(raw_svg.svg_string)
    primitives = extract_primitives(canonical_dom)
    document = infer_ir(primitives)
    if not document.nodes:
        return IRLabelResult(accepted=False, reason="empty_ir_document")

    confidence = score_ir(document)
    if confidence.score < 0.35:
        return IRLabelResult(accepted=False, reason="low_confidence_ir")

    rendered_svg = render_ir(document)
    if not roundtrip_ok(canonical_dom.original_svg, rendered_svg):
        return IRLabelResult(accepted=False, reason="roundtrip_failed")

    raw_svg.metadata = dict(raw_svg.metadata)
    raw_svg.metadata["diagram_ir"] = document.to_dict()
    raw_svg.metadata["diagram_ir_schema_version"] = document.schema_version
    raw_svg.metadata["diagram_ir_metrics"] = confidence.metrics
    raw_svg.metadata["diagram_ir_confidence"] = confidence.score
    raw_svg.metadata["diagram_ir_accepted"] = True
    raw_svg.metadata["diagram_ir_source_svg"] = canonical_dom.original_svg
    raw_svg.metadata["diagram_ir_rendered_svg"] = rendered_svg
    return IRLabelResult(accepted=True, reason="ok")
