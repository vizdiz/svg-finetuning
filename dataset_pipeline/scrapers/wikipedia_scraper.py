"""
dataset_pipeline.scrapers.wikipedia_scraper

Scrapes SVG files from Wikimedia Commons.

Queries the Wikimedia Commons API for files in the category tree rooted at
"SVG files" (e.g. SVG diagrams, SVG maps, SVG flags, SVG charts), downloads
each file's raw SVG content, and yields RawSVGRecord objects with the file's
page title, description, and Commons URL as metadata.

Uses the file description page to extract human-written captions and
categories, which are passed through to the captioner as prior context
to improve caption quality.

Handles:
  - Paginated API responses (cmcontinue)
  - Skipping non-SVG MIME types returned by the category API
  - Respecting Wikimedia's rate limits via httpx with backoff
"""
