"""
dataset_pipeline.processing.captioner

Generates natural-language prompts for each SVG using the Anthropic API.

Renders each normalised SVG to a PNG via cairosvg, then sends the image
to Claude with a structured prompt asking for a concise, imperative
description of what the SVG depicts (e.g. "A bar chart showing quarterly
revenue from 2020 to 2024"). Any existing metadata from the scraper
(Wikipedia caption, arXiv figure caption) is included as context to
improve accuracy and reduce hallucination.

Output is a single prompt string stored alongside the SVG in the
TrainingRecord — this becomes the model's input at training time.

Handles:
  - Anthropic API rate limiting and retries with exponential backoff
  - Captions that are too generic or too long (configurable bounds)
  - Batching requests to stay within API throughput limits
  - Caching responses by SVG content hash to avoid re-captioning duplicates
    across pipeline runs
"""
