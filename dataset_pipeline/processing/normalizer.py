"""
dataset_pipeline.processing.normalizer

Canonicalises SVG markup to reduce surface variation in the training data.

Transformations applied:
  - Strip XML declarations, comments, and processing instructions
  - Remove editor metadata (Inkscape, Illustrator, Sketch namespaces)
  - Normalise attribute ordering (alphabetical within each element)
  - Collapse redundant group (<g>) elements with no attributes
  - Convert style="" attribute shorthand to explicit presentation attributes
    where straightforward (fill, stroke, stroke-width, opacity)
  - Round floating-point coordinates to a configurable number of decimal places
  - Ensure a viewBox is present; derive from width/height if missing
  - Re-serialise with consistent indentation via lxml

Does not alter visual appearance — purely a textual canonicalisation so the
model learns from content rather than formatting variation.
"""
