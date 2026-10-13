"""Independent public-API reference checks; ArviZ is development-only."""

import json
from pathlib import Path

import numpy as np
import pytest

from posterior_lab.diagnostics import diagnose, rank_normalize, split_chains


def fixtures():
    rng = np.random.default_rng(20261006)
    independent = rng.normal(size=(4, 501))
    autocorrelated = np.zeros((4, 500))
    innovations = rng.normal(size=autocorrelated.shape)
    for index in range(1, 500):
        autocorrelated[:, index] = 0.9 * autocorrelated[:, index - 1] + innovations[:, index]
    shifted = independent + np.arange(4)[:, None] * 0.8
    unequal = independent * np.array([0.5, 1, 2, 4])[:, None]
    return {"independent_odd": independent, "independent_even": independent[:, :-1],
            "autocorrelated": autocorrelated, "shifted": shifted,
            "unequal_scales": unequal, "ties": np.round(independent, 1),
            "heavy_tail": np.exp(independent * 1.8)}


@pytest.mark.parametrize("name", list(fixtures()))
def test_saved_arviz_reference_fixtures(name):
    reference = json.loads((Path(__file__).parent / "fixtures" / "diagnostics-arviz-0.21.0.json").read_text())
    assert reference["reference_version"] == "0.21.0"
    actual = diagnose(fixtures()[name][:, :, None], ["x"])["parameters"][0]
    for metric, expected in reference["fixtures"][name].items():
        assert actual[metric] == pytest.approx(expected, rel=2e-10, abs=2e-10), metric


@pytest.mark.parametrize("name", list(fixtures()))
def test_matches_arviz_public_reference(name):
    az = pytest.importorskip("arviz")
    assert az.__version__ == "0.21.0", "Reference version must remain pinned"
    values = fixtures()[name]
    actual = diagnose(values[:, :, None], ["x"])["parameters"][0]
    expected = {"rank_rhat": az.rhat(values, method="z_scale"),
                "folded_rhat": az.rhat(values, method="folded"),
                "rhat": az.rhat(values, method="rank"),
                "ess_bulk": az.ess(values, method="bulk"),
                "ess_tail": az.ess(values, method="tail"),
                "ess_mean": az.ess(values, method="mean"),
                "mcse_mean": az.mcse(values, method="mean")}
    for metric, reference in expected.items():
        assert actual[metric] == pytest.approx(float(reference), rel=2e-10, abs=2e-10), metric


def test_odd_middle_discarded_only_for_split_diagnostics():
    values = np.arange(14, dtype=float).reshape(2, 7)
    expected = np.array([[0, 1, 2], [7, 8, 9], [4, 5, 6], [11, 12, 13]])
    np.testing.assert_array_equal(split_chains(values), expected)
    altered = values.copy()
    altered[:, 3] = 100
    report = diagnose(altered[:, :, None])["parameters"][0]
    assert report["mean"] == altered.mean()
    assert report["rhat"] == diagnose(values[:, :, None])["parameters"][0]["rhat"]


def test_average_ranks_for_ties():
    values = np.array([[1, 2, 2], [3, 1, 5]], dtype=float)
    ranks = rank_normalize(values)
    assert ranks[0, 0] == ranks[1, 1]
    assert ranks[0, 1] == ranks[0, 2]
    assert ranks[1, 2] > ranks[1, 0] > ranks[0, 1] > ranks[0, 0]


@pytest.mark.parametrize("kind", ["constant", "one_constant_chain", "nonfinite", "too_short", "one_chain", "empty"])
def test_unavailable_is_explicit_and_strict_json_safe(kind):
    values = np.random.default_rng(4).normal(size=(4, 100, 1))
    if kind == "constant":
        values[:] = 2
    elif kind == "one_constant_chain":
        values[0] = 2
    elif kind == "nonfinite":
        values[0, 0, 0] = np.inf
    elif kind == "too_short":
        values = values[:, :3]
    elif kind == "one_chain":
        values = values[:1]
    else:
        values = values[:, :0]
    result = diagnose(values)
    assert result["status"] == "unavailable"
    assert result["parameters"][0]["rhat"] is None
    assert result["parameters"][0]["mcse_mean"] is None
    json.dumps(result, allow_nan=False)


def test_constant_arviz_difference_is_intentional():
    az = pytest.importorskip("arviz")
    values = np.ones((4, 100))
    # ArviZ's constant ESS is N; Posterior follows posterior's safer policy.
    assert float(az.ess(values, method="bulk")) == values.size
    assert diagnose(values[:, :, None])["parameters"][0]["ess_bulk"] is None


def test_mean_mcse_uses_raw_scale_ess():
    values = fixtures()["heavy_tail"]
    row = diagnose(values[:, :, None])["parameters"][0]
    assert row["mcse_mean"] == pytest.approx(values.std(ddof=1) / np.sqrt(row["ess_mean"]))
    assert abs(row["ess_mean"] - row["ess_bulk"]) > 5


def test_anticorrelation_ess_can_exceed_draw_count():
    rng = np.random.default_rng(66)
    values = np.zeros((4, 1000))
    for index in range(1, 1000):
        values[:, index] = -0.7 * values[:, index - 1] + rng.normal(size=4)
    assert diagnose(values[:, :, None])["parameters"][0]["ess_mean"] > values.size


def test_disagreement_and_scale_differences_are_detected():
    examples = fixtures()
    shifted = diagnose(examples["shifted"][:, :, None])
    scales = diagnose(examples["unequal_scales"][:, :, None])["parameters"][0]
    assert shifted["message"] == "chains disagree"
    assert scales["folded_rhat"] > scales["rank_rhat"]
    assert scales["rhat"] > 1.1


def test_input_validation_and_minimum_finite_case():
    with pytest.raises(ValueError, match="shape"):
        diagnose(np.zeros((4, 10)))
    with pytest.raises(ValueError, match="names"):
        diagnose(np.zeros((4, 10, 2)), ["x", "x"])
    with pytest.raises(ValueError, match="limits"):
        diagnose(np.zeros((5, 10, 1)))
    result = diagnose(np.random.default_rng(12).normal(size=(2, 4, 1)))
    assert result["parameters"][0]["rhat"] is not None
    json.dumps(result, allow_nan=False)
