"""Hermetic test profile: BEML_ENV=test marks the suite as an explicitly
non-production profile so production boot gates (Keycloak coordinates,
PostgreSQL-backed MLflow tracking) do not fire; the gates themselves are
covered by dedicated tests that set the production profile explicitly."""

import os

os.environ.setdefault("BEML_ENV", "test")
