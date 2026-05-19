"""
dataset_pipeline.scrapers.arxiv_scraper

Scrapes SVG figures from arXiv papers.

Uses the `arxiv` Python client to query papers by category and date range,
downloads each paper's source tarball via the arXiv S3 bulk data bucket,
extracts embedded SVG files and figures converted from PDF pages using PyMuPDF,
and yields RawSVGRecord objects.

Targeted categories: cs.CV, cs.LG, cs.GR (computer vision, ML, graphics) where
SVG figures (diagrams, plots, architecture charts) are most prevalent.

Handles:
  - Deduplication by arXiv paper ID + figure index
  - Skipping papers whose source tarballs are unavailable
  - Rate limiting to respect arXiv bulk access guidelines
"""
