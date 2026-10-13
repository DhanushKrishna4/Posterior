"""Independent integrator identities and controlled-random Algorithm 6 fixtures."""

import copy
import json
import math

import numpy as np
import pytest

from posterior_lab.samplers import (
    leapfrog, hmc_transition, nuts_transition, _State, _Context, _build_tree,
    _recursive_probability, _top_level_probability,
)
from posterior_lab.adaptation import (
    coordinate_target, WarmupAdapter, WelfordVariance, find_reasonable_step_size,
    DualAveraging,
)


def normal(q):
    return -0.5 * float(q @ q), -q


def flat(q):
    return 0.0, np.zeros_like(q)


class FixedRNG:
    def __init__(self, uniforms, momentum=1.0):
        self.uniforms = iter(uniforms)
        self.momentum = momentum

    def normal(self, size):
        return np.full(size, self.momentum)

    def random(self):
        return next(self.uniforms)


def test_leapfrog_analytical_and_reversible():
    q, r = np.array([0.7, -0.2]), np.array([-0.3, 0.9])
    q0, r0 = q.copy(), r.copy()
    eps = 0.17
    expected_half = r - eps * q / 2
    expected_q = q + eps * expected_half
    expected_r = expected_half - eps * expected_q / 2
    q1, r1, lp, grad = leapfrog(normal, q, r, -q, eps)
    np.testing.assert_allclose(q1, expected_q, rtol=0, atol=1e-15)
    np.testing.assert_allclose(r1, expected_r, rtol=0, atol=1e-15)
    q2, r2, _, _ = leapfrog(normal, q1, r1, grad, -eps)
    np.testing.assert_allclose(q2, q, atol=1e-15)
    np.testing.assert_allclose(r2, r, atol=1e-15)
    q3, r3, _, _ = leapfrog(normal, q1, -r1, grad, eps)
    np.testing.assert_allclose(q3, q, atol=1e-15)
    np.testing.assert_allclose(r3, -r, atol=1e-15)
    np.testing.assert_array_equal(q, q0)
    np.testing.assert_array_equal(r, r0)


def test_energy_error_decreases_quadratically_at_fixed_integration_time():
    errors = []
    for eps, steps in [(0.2, 5), (0.1, 10), (0.05, 20)]:
        q, r = np.array([0.6]), np.array([0.8])
        g, energy0 = -q, 0.5
        for _ in range(steps):
            q, r, lp, g = leapfrog(normal, q, r, g, eps)
        errors.append(abs(-lp + float(r @ r) / 2 - energy0))
    assert 3.5 < errors[0] / errors[1] < 4.5
    assert 3.5 < errors[1] / errors[2] < 4.5


def test_hmc_acceptance_momentum_flip_and_repeated_rejection():
    q = np.array([0.0])
    result, meta = hmc_transition(normal, q, FixedRNG([0.8]), 2.5, steps=1, record=True)
    # q'=2.5, r'=1 - 2.5*2.5/2 = -2.125 before proposal flip.
    error = 0.5 * (2.5**2 + 2.125**2 - 1)
    assert meta["energy_error"] == pytest.approx(error)
    assert meta["acceptance"] == pytest.approx(math.exp(-error))
    assert meta["proposed_momentum"] == [2.125]
    assert meta["accepted"] is False
    np.testing.assert_array_equal(result, q)
    assert meta["gradient_evaluations"] == 2
    assert meta["integration_steps"] == 1
    accepted, yes = hmc_transition(normal, q, FixedRNG([0.00001]), 2.5, steps=1)
    assert yes["accepted"]
    np.testing.assert_array_equal(accepted, [2.5])


@pytest.mark.parametrize("direction", [-1, 1])
def test_nuts_exact_small_tree_and_top_level_rule(direction):
    # Free-particle fixture. At depth 1 recursive selection probability is 1/2,
    # but top-level probability is min(1,2/2)=1, not 2/(2+2)=1/2.
    d = 0.1 if direction == -1 else 0.9
    rng = FixedRNG([0.5, d, 0.7, d, 0.75, 0.75])
    q, meta = nuts_transition(flat, [0.0], rng, 1.0, max_depth=2, record=True)
    np.testing.assert_array_equal(q, [2.0 * direction])
    assert meta["candidate_count"] == 4
    assert meta["integration_steps"] == 3
    assert meta["gradient_evaluations"] == 4
    assert meta["depth"] == 2
    assert meta["max_depth_reached"] is True
    assert meta["divergent"] is False
    expansions = [v for v in meta["branches"] if v["kind"] == "expansion"]
    assert [v["selection_probability"] for v in expansions] == [1.0, 1.0]
    assert [p["position"][0] for p in meta["trajectory"]] == [0, direction, 2 * direction, 3 * direction]


def test_unequal_recursive_counts_are_not_equal_branch_probabilities():
    # This discontinuous synthetic target is solely a candidate-bookkeeping
    # oracle: its zero gradients produce exactly known integer locations.
    def fixture(q):
        return (-0.8 if q[0] == 3 else 0.0), np.zeros_like(q)

    context = _Context(fixture, None, True, -0.5)
    start = _State(np.array([0.0]), np.array([1.0]), 0.0, np.array([0.0]))
    tree = _build_tree(start, -1.0, 1, 2, 1.0, FixedRNG([0.9, 0.9, 0.4]), context)
    assert tree.count == 3
    np.testing.assert_array_equal(tree.candidate.q, [1.0])
    combined = context.branches[-1]
    assert combined["first_count"] == 2
    assert combined["second_count"] == 1
    assert combined["selection_probability"] == pytest.approx(1 / 3)
    assert _recursive_probability(0, 0) == 0
    assert _top_level_probability(2, 1) == 0.5


def test_internal_uturn_excludes_whole_new_subtree():
    context = _Context(normal, None, True, -0.5)
    start = _State(np.array([0.0]), np.array([1.0]), 0.0, np.array([0.0]))
    tree = _build_tree(start, -10.0, 1, 2, 1.0, FixedRNG([0.1, 0.1, 0.1]), context)
    assert tree.count == 4
    assert tree.continuing is False
    assert context.branches[-1]["internal_uturn"] is True
    assert context.divergent is False


def test_recorded_slice_eligible_leaves_of_stopped_subtree_are_excluded():
    # Harmonic oscillator at epsilon=1: first 1- and 2-leaf expansions pass;
    # the final 4-leaf expansion develops an internal U-turn. All leaves lie
    # inside the slice, but none of those final four may replace the candidate.
    rng = FixedRNG([0.0001, 0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
    q, metadata = nuts_transition(normal, [0.0], rng, 1.0, max_depth=3, record=True)
    np.testing.assert_array_equal(q, [0.0])
    assert metadata["stop_reason"] == "internal U-turn"
    assert not metadata["divergent"]
    assert not metadata["max_depth_reached"]
    stopped = metadata["trajectory"][4:]
    assert len(stopped) == 4
    assert all(p["slice_eligible"] and not p["eligible"] for p in stopped)


def test_slice_relative_continuation_distinct_from_energy_threshold():
    def fixture(q):
        return -1000.2, np.zeros_like(q)

    context = _Context(fixture, None, True, -0.5)
    start = _State(np.array([0.0]), np.array([1.0]), 0.0, np.array([0.0]))
    tree = _build_tree(start, -1.5, 1, 0, 1.0, FixedRNG([]), context)
    assert tree.continuing is True  # -1.5 < -1000.7 + 1000
    assert tree.count == 0
    assert context.max_energy_error == pytest.approx(1000.2)
    assert context.divergent is False


def test_invalid_subtree_is_never_selected():
    def invalid(q):
        return (0.0, np.zeros_like(q)) if q[0] == 0 else (-np.inf, np.array([np.nan]))

    result, meta = nuts_transition(invalid, [0.0], FixedRNG([0.5, 0.9]), 1.0, record=True)
    np.testing.assert_array_equal(result, [0.0])
    assert meta["divergent"] is True
    assert meta["max_depth_reached"] is False
    assert meta["acceptance"] == 0
    assert meta["branches"][0]["subtree_valid"] is False
    json.dumps(meta, allow_nan=False)


def test_nuts_adaptation_statistic_uses_only_final_doubling():
    rng = FixedRNG([0.1, 0.9, 0.2, 0.9, 0.2, 0.2])
    _, meta = nuts_transition(normal, [0.0], rng, 0.2, max_depth=2, record=True)
    errors = [p["energy_error"] for p in meta["trajectory"][1:]]
    probabilities = [min(1.0, math.exp(-e)) for e in errors]
    assert len(errors) == 3
    assert meta["acceptance"] == pytest.approx(np.mean(probabilities[1:]))
    assert meta["acceptance_all_leaves"] == pytest.approx(np.mean(probabilities))
    assert meta["acceptance"] != meta["acceptance_all_leaves"]


@pytest.mark.parametrize("sampler", [hmc_transition, nuts_transition])
def test_cancellation_propagates_without_mutating_position(sampler):
    class Cancelled(RuntimeError):
        pass

    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise Cancelled("budget exhausted")

    q = np.array([0.0])
    with pytest.raises(Cancelled):
        sampler(flat, q, np.random.default_rng(123), 0.1, check=check)
    np.testing.assert_array_equal(q, [0.0])
    assert calls == 4


def test_trajectory_recording_is_bounded_and_does_not_change_rng():
    first_rng = np.random.default_rng(33)
    second_rng = np.random.default_rng(33)
    a, detailed = hmc_transition(flat, [0.0], first_rng, 0.01, steps=300, record=True)
    b, compact = hmc_transition(flat, [0.0], second_rng, 0.01, steps=300, record=False)
    np.testing.assert_array_equal(a, b)
    assert first_rng.bit_generator.state == second_rng.bit_generator.state
    assert len(detailed["trajectory"]) == 256
    assert detailed["omitted_trajectory_points"] == 45
    assert compact["trajectory"] == []


def test_deep_nuts_record_retains_every_expansion_with_bounded_branches():
    import itertools

    _, metadata = nuts_transition(flat, [0.0], FixedRNG(itertools.repeat(0.9)),
                                  0.01, max_depth=9, record=True)
    expansions = [b for b in metadata["branches"] if b["kind"] == "expansion"]
    assert [b["depth"] for b in expansions] == list(range(9))
    assert metadata["integration_steps"] == 511
    assert len(metadata["trajectory"]) == 256
    assert metadata["omitted_trajectory_points"] == 256
    assert len(metadata["branches"]) == 128
    assert metadata["omitted_branches"] == 383


def test_affine_density_gradient_and_fixed_scale_capture():
    scale = np.array([0.1, 3.0])
    working = coordinate_target(normal, scale)
    x = np.array([2.0, -0.7])
    lp, grad = working(x)
    np.testing.assert_allclose(grad, -(scale**2) * x)
    assert lp == pytest.approx(normal(scale * x)[0] + math.log(0.3))
    finite_difference = []
    for i in range(2):
        direction = np.eye(2)[i] * 1e-6
        finite_difference.append((working(x + direction)[0] - working(x - direction)[0]) / 2e-6)
    np.testing.assert_allclose(grad, finite_difference, rtol=1e-8)
    scale[:] = 100  # An external warmup update cannot modify this transition.
    np.testing.assert_array_equal(working(x)[1], grad)


def test_welford_matches_two_pass_variance_and_bounds():
    data = np.random.default_rng(3).normal(size=(80, 2)) * [0.03, 20] + [1000, -900]
    acc = WelfordVariance(2)
    for row in data:
        acc.update(row)
    expected = 80 / 85 * data.var(axis=0, ddof=1) + 5 / 85 * 1e-3
    np.testing.assert_allclose(acc.regularized_variance(), expected, rtol=1e-10)
    same = WelfordVariance(2)
    same.update([1, 1])
    same.update([1, 1])
    assert np.all(same.regularized_variance() > 0)


def test_warmup_boundaries_freeze_and_exact_checkpoint_continuation():
    adapter = WarmupAdapter(2, 120)
    data = np.random.default_rng(43).normal(size=(120, 2)) * [0.3, 3.0]
    for row in data[:43]:
        adapter.update(row, 0.8)
    restored = WarmupAdapter.from_state(json.loads(json.dumps(adapter.state_dict())))
    boundaries = []
    for row in data[43:]:
        boundaries.append(adapter.update(row, 0.75))
        restored.update(row, 0.75)
    assert any(boundaries)
    assert adapter.state_dict() == restored.state_dict()
    assert adapter.stage == "frozen"
    assert adapter.scales[0] < 1 < adapter.scales[1]
    frozen = copy.deepcopy(adapter.state_dict())
    assert adapter.update([999, 999], 0.01) is False
    assert adapter.state_dict() == frozen
    short = WarmupAdapter(2, 12)
    for row in data[:12]:
        short.update(row, 0.8)
    np.testing.assert_array_equal(short.scales, [1.0, 1.0])
    assert "short warmup" in short.status


def test_bounded_step_search_counts_work_and_handles_invalid_trials():
    def narrow(q):
        return -0.5 * float(q @ q) / 1e-6, -q / 1e-6

    step, info = find_reasonable_step_size(narrow, [0.0], np.random.default_rng(5),
                                          initial=1.0, max_trials=20, return_info=True)
    assert 1e-5 <= step <= 0.01
    assert info["gradient_evaluations"] <= 21
    assert info["status"] == "crossed 0.5 acceptance"


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(dim=True),
    lambda s: s.update(iteration=1.5),
    lambda s: s.update(scales=[1.0] * 33),
    lambda s: s.update(scales=[0.0, 1.0]),
    lambda s: s.update(scales=[2.0, 1.0]),
    lambda s: s.update(step_size=0.03),
    lambda s: s.update(status="frozen"),
    lambda s: s["dual"].update(iteration=41),
    lambda s: s["dual"].update(target=0.9),
    lambda s: s["dual"].update(hbar=float("nan")),
    lambda s: s["dual"].update(log_step_bar=float("inf")),
    lambda s: s["variance"].update(count=-1),
    lambda s: s["variance"].update(count=9),
    lambda s: s["variance"].update(count=10.0),
    lambda s: s["variance"].update(mean=[0.0]),
    lambda s: s["variance"].update(m2=[-1.0, 0.0]),
    lambda s: s["variance"].update(m2=[float("inf"), 0.0]),
])
def test_adaptation_checkpoint_rejects_invalid_or_inconsistent_state(mutation):
    adapter = WarmupAdapter(2, 200)
    for i in range(40):
        adapter.update([math.sin(i), math.cos(i)], 0.8)
    state = adapter.state_dict()
    mutation(state)
    with pytest.raises(ValueError):
        WarmupAdapter.from_state(state)


def test_welford_import_preflights_shapes_before_numpy_allocation(monkeypatch):
    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("NumPy must not see an oversized state vector")

    monkeypatch.setattr(np, "asarray", unexpected_allocation)
    with pytest.raises(ValueError, match="vector"):
        WelfordVariance.from_state({"count": 0, "mean": [0.0] * 10000, "m2": []})


def test_empty_welford_and_restarted_dual_checkpoint_consistency():
    with pytest.raises(ValueError, match="empty"):
        WelfordVariance.from_state({"count": 0, "mean": [1.0], "m2": [0.0]})
    with pytest.raises(ValueError, match="single"):
        WelfordVariance.from_state({"count": 1, "mean": [1.0], "m2": [1.0]})
    state = WarmupAdapter(1, 50).state_dict()["dual"]
    state["hbar"] = 0.1
    with pytest.raises(ValueError, match="restarted"):
        DualAveraging.from_state(state)


@pytest.mark.parametrize("sampler", [hmc_transition, nuts_transition])
def test_prespecified_seed_gaussian_moments(sampler):
    # Three fixed seeds; no retry policy. These aggregate tolerances correspond
    # to roughly 5 ordinary standard errors for >=400 effective scalar draws.
    rows = []
    for seed in [17, 29, 53]:
        rng, q = np.random.default_rng(seed), np.zeros(2)
        for i in range(700):
            q, _ = sampler(normal, q, rng, 0.35)
            if i >= 100:
                rows.append(q)
    rows = np.asarray(rows)
    np.testing.assert_allclose(rows.mean(axis=0), [0, 0], atol=0.17)
    np.testing.assert_allclose(np.cov(rows.T), np.eye(2), atol=0.22)


@pytest.mark.parametrize("sampler", [hmc_transition, nuts_transition])
def test_beta_binomial_transformed_target_mean_and_quantiles(sampler):
    from scipy.special import expit
    from scipy.stats import beta

    def target(q):
        z = q[0]
        # Beta(9,5) posterior in logit coordinates, including its Jacobian.
        return -9 * np.logaddexp(0, -z) - 5 * np.logaddexp(0, z), np.array([9 - 14 * expit(z)])

    draws = []
    for seed in [17, 29, 53]:
        q, rng = np.zeros(1), np.random.default_rng(seed)
        for i in range(700):
            # Ten steps at .35 resonate with this target's local logit period
            # (~3.5). Half that integration time avoids the verified resonance.
            settings = {"steps": 5} if sampler is hmc_transition else {}
            q, _ = sampler(target, q, rng, 0.35, **settings)
            if i >= 100:
                draws.append(expit(q[0]))
    assert np.mean(draws) == pytest.approx(9 / 14, abs=0.025)
    np.testing.assert_allclose(np.quantile(draws, [0.1, 0.5, 0.9]), beta.ppf([0.1, 0.5, 0.9], 9, 5), atol=0.035)


@pytest.mark.parametrize("name", ["beta", "gaussian", "regression"])
@pytest.mark.parametrize("sampler", [hmc_transition, nuts_transition])
def test_compiled_graph_distributional_validation(name, sampler):
    """Actual preset graphs, independent oracles, fixed seeds and quality gates.

    Four chains each retain 750 draws after 100 burn-in transitions. Acceptance
    alone is not a quality gate: require rank R-hat <1.01 and >=400 bulk ESS.
    Mean errors must be below five raw-mean MCSEs; covariance error below .2 in
    standardized units and marginal quantile error below .25 marginal SDs.
    These broad statistical tolerances do not establish performance guarantees.
    """
    from scipy.stats import beta, norm
    from posterior_lab.examples import preset, reference
    from posterior_lab.models import compile_model
    from posterior_lab.diagnostics import diagnose

    spec = preset(name)["spec"]
    model = compile_model(spec)
    oracle = reference(spec)
    mean = np.asarray(oracle["mean"])
    covariance = (np.diag(oracle["variance"]) if name == "beta"
                  else np.asarray(oracle["covariance"]))
    sd = np.sqrt(np.diag(covariance))
    # Conjugate regression's known scale makes this also an affine-coordinate
    # sampler identity check. Adaptive scales are verified in separate tests.
    scale = sd if name == "regression" else np.ones(model.dimension)
    target = coordinate_target(model, scale)
    chains = []
    for seed in [17, 29, 53, 101]:
        q, rng, rows = np.zeros(model.dimension), np.random.default_rng(seed), []
        for i in range(850):
            kwargs = {"steps": 5} if sampler is hmc_transition else {}
            q, metadata = sampler(target, q, rng, 0.2, **kwargs)
            assert not metadata["divergent"]
            if i >= 100:
                rows.append(model.report(q * scale))
        chains.append(rows)
    chains = np.asarray(chains)
    diagnostics = diagnose(chains)["parameters"]
    for row in diagnostics:
        assert row["rhat"] < 1.01
        assert row["ess_bulk"] > 400
    flat_draws = chains.reshape(-1, model.dimension)
    mcse = np.array([row["mcse_mean"] for row in diagnostics])
    assert np.all(abs(flat_draws.mean(axis=0) - mean) < 5 * mcse)
    observed_cov = np.atleast_2d(np.cov(flat_draws.T))
    np.testing.assert_allclose(observed_cov / np.outer(sd, sd),
                               covariance / np.outer(sd, sd), atol=0.2)
    quantile_levels = np.array([0.05, 0.5, 0.95])
    exact_quantiles = (beta.ppf(quantile_levels, 9, 5)[:, None] if name == "beta"
                       else mean + norm.ppf(quantile_levels)[:, None] * sd)
    observed_quantiles = np.quantile(flat_draws, quantile_levels, axis=0)
    assert np.all(abs(observed_quantiles - exact_quantiles) < 0.25 * sd)
