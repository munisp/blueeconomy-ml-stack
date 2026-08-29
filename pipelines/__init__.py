"""Lakehouse integration + continuous training pipelines.

Reads from the platform lakehouse pattern established by
blueeconomy-data-platform (env-driven storage roots, medallion layers,
Parquet/GeoParquet datasets with lineage). When production volume is below a
configurable threshold the pipelines fall back to the SYNTHETIC generators —
clearly labelled — so the training loop is exercisable before real data flows.
"""
