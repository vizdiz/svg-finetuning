"""
dataset_pipeline.pipeline.manifest_writer

Finalises a pipeline run by writing a DatasetManifest to S3.

Collects the S3 URIs of all uploaded JSONL files for the current batch,
constructs a DatasetManifest (using the schema from training/dataset_interface.py),
and writes it to s3://<data-bucket>/train/dataset_manifest.json.

Writing the manifest is the final step and the only action that triggers
the training Lambda — the manifest must not be written until all JSONL
files are fully uploaded and verified.

Also writes a timestamped archive copy to
s3://<data-bucket>/train/manifests/<batch-id>.json so previous runs are
preserved and can be replayed.
"""
