"""A/B deterministic split tests."""

import pytest

from monitoring.ab import HashSplitter


def test_deterministic_routing():
    s = HashSplitter(["0.1.0", "0.2.0"], [0.5, 0.5])
    routes = [s.route(f"entity-{i}") for i in range(500)]
    assert routes == [s.route(f"entity-{i}") for i in range(500)]  # stable
    assert set(routes) <= {"0.1.0", "0.2.0"}
    # both buckets get traffic
    assert 0.3 < routes.count("0.1.0") / len(routes) < 0.7


def test_weight_validation():
    with pytest.raises(ValueError):
        HashSplitter(["a"], [0.0])
    with pytest.raises(ValueError):
        HashSplitter(["a", "b"], [1.0])
