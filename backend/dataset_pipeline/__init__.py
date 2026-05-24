"""Corpus-first dataset pipeline for SVG/diagram fine-tuning.

The old live scraper orchestration path has been removed. Dataset builds now
flow through normalized corpus candidates, source-specific fetch/extract
workers, validation, dedupe, normalization, product assembly, and an explicit
manual promotion gate before `train/dataset_manifest.json` is written.
"""
