"""VI checks against exact Gaussian KL and independent scalar quadrature."""

import json
from copy import deepcopy
import shutil
import subprocess

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import betaln, expit

from posterior_lab.variational import MeanFieldVI, VIFailure, fit_vi


def gaussian(mean, covariance):
    mean = np.asarray(mean, dtype=np.float64)
    precision = np.linalg.inv(covariance)
    normalization = len(mean) * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1]

    def target(value):
        difference = value - mean
        return float(-0.5 * (normalization + difference @ precision @ difference)), -precision @ difference

    return target


def gaussian_kl(mu, variance, target_mean, target_covariance):
    precision = np.linalg.inv(target_covariance)
    difference = mu - target_mean
    return 0.5 * (np.sum(np.diag(precision) * variance) + difference @ precision @ difference
                  - len(mu) + np.linalg.slogdet(target_covariance)[1] - np.log(variance).sum())


@pytest.mark.parametrize("seed", [19, 41, 73])
def test_reverse_kl_gaussian_optimum_and_objective(seed):
    # Declared statistical tolerances cover fresh batch noise at a fixed budget.
    # All three fixed seeds must pass: never search for a favorable seed.
    mean = np.array([0.8, -0.4])
    covariance = np.array([[1, 0.95], [0.95, 1]])
    target = gaussian(mean, covariance)
    vi = fit_vi(target, 2, iterations=2000, seed=seed, init=[-0.5, 0.8],
                learning_rate=0.015, batch_size=64, evaluation_interval=500)
    optimum_variance = 1 / np.diag(np.linalg.inv(covariance))
    np.testing.assert_allclose(optimum_variance, [0.0975, 0.0975], atol=1e-14)
    np.testing.assert_allclose(vi.mu, mean, atol=0.085, rtol=0)
    np.testing.assert_allclose(vi.scale**2, optimum_variance, rtol=0.16, atol=0)
    actual_kl = gaussian_kl(vi.mu, vi.scale**2, mean, covariance)
    optimal_kl = gaussian_kl(mean, optimum_variance, mean, covariance)
    assert 0 <= actual_kl - optimal_kl < 0.06
    evaluation = vi.evaluate(20000)
    assert abs(evaluation["elbo"] + actual_kl) < 5 * evaluation["mcse"]
    assert len(vi.history) == 2000
    assert vi.gradient_evaluations == 2000 * 64
    assert vi.status == "complete"
    assert any(np.diff([point["optimization_elbo"] for point in vi.history]) < 0)


def test_reparameterization_gradient_matches_fixed_noise_finite_difference():
    target = gaussian([0.3, -0.1], [[2, 0.5], [0.5, 1]])
    vi = MeanFieldVI(target, 2, seed=96, init={"mu": [1, -1], "rho": [-0.3, 0.2]}, batch_size=13)
    rng = np.random.default_rng()
    rng.bit_generator.state = vi.rng.bit_generator.state
    epsilon = rng.normal(size=(13, 2))
    parameters = np.concatenate((vi.mu, vi.rho))

    def objective(values):
        mu, rho = np.split(values, 2)
        density = np.mean([target(mu + np.exp(rho) * draw)[0] for draw in epsilon])
        return density + rho.sum() + 1 + np.log(2 * np.pi)

    difference_gradient = np.empty(4)
    for index in range(4):
        offset = np.zeros(4)
        offset[index] = 1e-6
        difference_gradient[index] = (objective(parameters + offset) - objective(parameters - offset)) / 2e-6
    metadata = vi.step()
    np.testing.assert_allclose(vi.first_moment / 0.1, difference_gradient, atol=2e-9, rtol=2e-9)
    assert metadata["optimization_elbo"] == pytest.approx(objective(parameters))


def test_transformed_beta_elbo_against_independent_quadrature():
    def target(z):
        value = z[0]
        logp = -9 * np.logaddexp(0, -value) - 5 * np.logaddexp(0, value) - betaln(9, 5)
        return float(logp), np.array([9 - 14 * expit(value)])

    vi = MeanFieldVI(target, 1, seed=77, init={"mu": [0.6], "rho": [-0.5]})
    mean, scale = vi.mu[0], vi.scale[0]

    def integrand(epsilon):
        return target(np.array([mean + scale * epsilon]))[0] * np.exp(-epsilon**2 / 2) / np.sqrt(2 * np.pi)

    integral, quadrature_error = quad(integrand, -12, 12, epsabs=1e-11)
    assert quadrature_error < 1e-9
    expected = integral + vi.entropy
    estimate = vi.evaluate(20000)
    assert abs(estimate["elbo"] - expected) < 5 * estimate["mcse"]


def test_pause_resume_exact_in_same_runtime_and_stream_independence():
    target = gaussian([0, 0], [[1, 0.5], [0.5, 2]])
    full = MeanFieldVI(target, 2, seed=9, max_iterations=60)
    observed = MeanFieldVI(target, 2, seed=9, max_iterations=60)
    for index in range(30):
        full.step()
        observed.step()
        if index % 10 == 0:
            observed.evaluate(32)
            observed.sample(13)
    np.testing.assert_array_equal(observed.mu, full.mu)
    checkpoint = json.loads(json.dumps(observed.state_dict(), allow_nan=False))
    restored = MeanFieldVI.from_state(target, checkpoint)
    for _ in range(30):
        full.step()
        observed.step()
        restored.step()
    np.testing.assert_array_equal(restored.mu, full.mu)
    np.testing.assert_array_equal(restored.rho, full.rho)
    assert restored.state_dict() == observed.state_dict()
    assert restored.evaluate(50) == observed.evaluate(50)
    np.testing.assert_array_equal(restored.sample(20), observed.sample(20))


def test_nonfinite_failure_is_transactional_and_not_clipped():
    def broken(z):
        return -np.inf, np.full_like(z, np.nan)

    vi = MeanFieldVI(broken, 2, seed=5)
    before = vi.state_dict()
    with pytest.raises(VIFailure, match="nonfinite"):
        vi.step()
    after = vi.state_dict()
    for field in ("mu", "rho", "rng", "first_moment", "second_moment", "iteration", "history"):
        assert before[field] == after[field]
    assert after["status"] == "failed"
    assert after["failure_reason"]
    json.dumps(after, allow_nan=False)


def test_scale_limits_iteration_limits_and_initialization():
    target = gaussian([0], [[1]])
    with pytest.raises(VIFailure, match="collapse"):
        MeanFieldVI(target, 1, init={"rho": [-15]})
    with pytest.raises(VIFailure, match="exceeded"):
        MeanFieldVI(target, 1, init={"rho": [9]})
    with pytest.raises(ValueError, match="dimension"):
        MeanFieldVI(target, 33)
    vi = MeanFieldVI(target, 1, max_iterations=1)
    vi.step()
    with pytest.raises(StopIteration):
        vi.step()
    different = MeanFieldVI(target, 1, seed=43)
    assert different.mu[0] != MeanFieldVI(target, 1, seed=42).mu[0]


def test_checkpoint_validation():
    target = gaussian([0], [[1]])
    state = MeanFieldVI(target, 1).state_dict()
    state["second_moment"] = [-1, 0]
    with pytest.raises(ValueError, match="negative"):
        MeanFieldVI.from_state(target, state)


@pytest.mark.parametrize("mutation", ["dimension", "nested_vector", "extra_field", "counter", "history", "history_nonfinite", "evaluation_count", "status", "rng", "boolean"])
def test_checkpoint_rejects_inconsistent_or_oversized_data(mutation):
    target = gaussian([0], [[1]])
    vi = MeanFieldVI(target, 1)
    vi.step()
    vi.evaluate(8)
    state = deepcopy(vi.state_dict())
    if mutation == "dimension":
        state["dimension"] = 33
    elif mutation == "nested_vector":
        state["mu"] = [[1] * 10000]
    elif mutation == "extra_field":
        state["execute"] = "never"
    elif mutation == "counter":
        state["gradient_evaluations"] += 1
    elif mutation == "history":
        state["history"][0]["iteration"] = 7
    elif mutation == "history_nonfinite":
        state["history"][0]["optimization_elbo"] = float("nan")
    elif mutation == "evaluation_count":
        state["evaluation_gradient_evaluations"] = 0
    elif mutation == "status":
        state["status"] = "complete"
    elif mutation == "rng":
        state["rng"]["state"]["inc"] = 2
    elif mutation == "boolean":
        state["mu"] = [True]
    with pytest.raises(ValueError):
        MeanFieldVI.from_state(target, state)


def test_checkpoint_requires_compiled_target_dimension_match():
    class Target:
        dimension = 2

        def __call__(self, z):
            return -0.5 * z @ z, -z

    state = MeanFieldVI(gaussian([0], [[1]]), 1).state_dict()
    with pytest.raises(ValueError, match="differs from target"):
        MeanFieldVI.from_state(Target(), state)


def test_internal_clone_does_not_commit_to_original():
    target = gaussian([0], [[1]])
    vi = MeanFieldVI(target, 1)
    vi.step()
    before = vi.state_dict()
    copied = vi.clone(target)
    copied.step()
    copied.evaluate(8)
    copied.sample(10)
    assert vi.state_dict() == before
    vi.step()
    vi.evaluate(8)
    vi.sample(10)
    assert copied.state_dict() == vi.state_dict()


def test_checkpoint_rng_state_uses_lossless_decimal_strings():
    target = gaussian([0, 0], [[1, 0.5], [0.5, 2]])
    vi = MeanFieldVI(target, 2, seed=67)
    vi.step()
    vi.evaluate(8)
    vi.sample(5)
    exported = vi.state_dict()
    for name in ("rng", "evaluation_rng", "sample_rng"):
        internal = getattr(vi, name).bit_generator.state
        for field in ("state", "inc"):
            encoded = exported[name]["state"][field]
            assert isinstance(encoded, str)
            assert encoded.isascii() and encoded.isdecimal()
            assert int(encoded) == internal["state"][field]
            # Demonstrate why exporting this as a JavaScript Number is unsafe.
            assert internal["state"][field] > 2**53
    # Old native JSON files with exact Python integers remain readable.
    legacy = deepcopy(exported)
    for name in ("rng", "evaluation_rng", "sample_rng"):
        legacy[name] = deepcopy(getattr(vi, name).bit_generator.state)
    assert MeanFieldVI.from_state(target, legacy).state_dict() == exported


def test_checkpoint_actual_javascript_json_roundtrip_preserves_all_streams():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for actual JavaScript JSON.parse/stringify verification")
    target = gaussian([0, 0], [[1, 0.5], [0.5, 2]])
    original = MeanFieldVI(target, 2, seed=241)
    for _ in range(7):
        original.step()
    original.evaluate(11)
    original.sample(13)
    source = "let s=''; process.stdin.setEncoding('utf8'); process.stdin.on('data', c => s += c); process.stdin.on('end', () => process.stdout.write(JSON.stringify(JSON.parse(s))));"
    roundtrip = subprocess.run([node, "-e", source], input=json.dumps(original.state_dict(), allow_nan=False),
                               capture_output=True, text=True, check=True, timeout=10)
    restored = MeanFieldVI.from_state(target, json.loads(roundtrip.stdout))
    assert original.state_dict() == restored.state_dict()
    for _ in range(5):
        assert original.step() == restored.step()
    assert original.evaluate(19) == restored.evaluate(19)
    np.testing.assert_array_equal(original.sample(23), restored.sample(23))
    assert original.state_dict() == restored.state_dict()
