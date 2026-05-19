"""
dataset_pipeline.pipeline

Orchestrates the full scrape → process → upload → manifest flow.

runner.py coordinates scrapers and processors, uploading completed
TrainingRecords to S3. manifest_writer.py finalises the run by writing
a DatasetManifest, which triggers the training Lambda.
"""
