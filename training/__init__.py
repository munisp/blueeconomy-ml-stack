"""Real PyTorch training loops for the BlueEconomy ML stack.

Every loop: seeded/reproducible, MLflow-tracked (falls back to a local JSON
run log when MLflow is not installed), early stopping, checkpointing,
CPU-friendly defaults with optional --device cuda.
"""
