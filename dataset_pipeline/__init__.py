"""
dataset_pipeline

End-to-end pipeline for building SVG training datasets for Qwen2.5-7B fine-tuning.

Scrapes SVGs from arXiv and Wikipedia, captions them using the Anthropic API,
normalizes and validates the SVG markup, then writes a DatasetManifest to S3
that triggers the training Lambda.

Entry point: dataset_pipeline/pipeline/runner.py
"""
