"""
dataset_pipeline.processing

Transforms raw RawSVGRecord objects into validated, captioned TrainingRecords
ready for upload to S3.

Stage order:
  1. validator  — structural and size checks before any expensive work
  2. normalizer — canonicalises SVG markup
  3. captioner  — generates natural-language prompts via the Anthropic API
"""
