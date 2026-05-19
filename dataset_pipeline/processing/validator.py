"""
dataset_pipeline.processing.validator

Validates raw SVG content before it enters the processing pipeline.

Checks performed:
  - Well-formed XML (parseable by lxml)
  - Root element is <svg> with a valid namespace
  - File size within configured bounds (min/max bytes)
  - Contains at least one visible drawing element (path, rect, circle, etc.)
  - Does not contain embedded raster images (base64 <image> tags) beyond a
    configurable threshold — keeps the dataset SVG-native
  - No script tags (security and cleanliness)
  - Viewport / viewBox is present and non-degenerate

Returns a ValidationResult with a pass/fail flag and a rejection reason
string so the pipeline can log why records were dropped.
"""
