"""CPU inference: ONNX export, fail-closed scoring library, FastAPI service.

Doctrine: deterministic rules are the first line of defence; ML augments.
When a model file is missing or invalid the scorer reports
SCORING_UNAVAILABLE and the service degrades to rules-only mode. It NEVER
fabricates a score.
"""
