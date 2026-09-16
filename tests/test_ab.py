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


def test_serving_splitter_honors_config_salt(tmp_path):
    """M1 regression: a non-default salt in the A/B config must route
    identically in serving (Scorer) and offline analysis (from_config)."""
    cfg = tmp_path / "ab.yaml"
    cfg.write_text(
        "salt: phase19-nondefault-salt\n"
        "models:\n"
        "  declaration-fraud:\n"
        "    versions: ['0.1.0', '0.2.0']\n"
        "    weights: [0.5, 0.5]\n")
    offline = HashSplitter.from_config(cfg, "declaration-fraud")
    from inference.scoring import Scorer
    scorer = Scorer(tmp_path, "declaration-fraud", offline.versions,
                    split=offline.weights, salt=offline.salt)
    assert scorer.splitter.salt == "phase19-nondefault-salt"
    for i in range(50):
        entity = f"entity-{i}"
        assert scorer.splitter.route(entity) == offline.route(entity)
    # and the non-default salt actually buckets differently somewhere
    default = HashSplitter(offline.versions, offline.weights)
    assert any(offline.route(f"entity-{i}") != default.route(f"entity-{i}")
               for i in range(50))
