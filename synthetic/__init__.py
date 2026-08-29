"""SYNTHETIC data generators for the BlueEconomy ML stack.

Every dataset produced by this package is SYNTHETIC. It is statistically
modelled on public characteristics of Nigerian maritime trade (ports, HS
chapters, CVFF contribution rules) but contains NO real declarations, vessels,
accounts or persons. Outputs are labelled SYNTHETIC in every schema and are
intended as training/development data only. They must NEVER be wired to any
production scoring endpoint.
"""

__all__ = ["declarations", "ais", "cvff", "config"]
