"""Mean-field Gaussian reparameterization VI with analytic entropy and Adam.

The target supplies the transformed log joint, including model transform
Jacobians exactly once. See Kucukelbir et al. (2017), sections 2.4--2.6.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np

from .serialization import encode_rng_state, decode_rng_state

Target = Callable[[np.ndarray], tuple[float, np.ndarray]]
LOG_2PI = float(np.log(2 * np.pi))
MIN_RHO, MAX_RHO = -14.0, 8.0


class VIFailure(RuntimeError):
    """An update cannot be completed without invalid arithmetic or collapsed scale."""


def _integer(value, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _clone_rng(rng: np.random.Generator) -> np.random.Generator:
    clone = np.random.Generator(np.random.PCG64())
    clone.bit_generator.state = deepcopy(rng.bit_generator.state)
    return clone


def _checkpoint_number(value, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return float(value)


def _checkpoint_vector(values, length: int, name: str) -> list:
    # Validate the data-only shape before NumPy can allocate an imported array.
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"invalid VI {name} dimension")
    for value in values:
        _checkpoint_number(value, name)
    return values


def _checkpoint_rng(state: dict) -> dict:
    if not isinstance(state, dict) or set(state) != {"bit_generator", "state", "has_uint32", "uinteger"} or state["bit_generator"] != "PCG64":
        raise ValueError("VI requires a complete PCG64 random state")
    inner = state["state"]
    if not isinstance(inner, dict) or set(inner) != {"state", "inc"}:
        raise ValueError("invalid PCG64 internal state")
    _integer(inner["state"], "PCG64 state", 0, 2**128 - 1)
    _integer(inner["inc"], "PCG64 increment", 1, 2**128 - 1)
    if inner["inc"] % 2 != 1:
        raise ValueError("PCG64 increment must be odd")
    _integer(state["has_uint32"], "PCG64 cached flag", 0, 1)
    _integer(state["uinteger"], "PCG64 cached integer", 0, 2**32 - 1)
    return deepcopy(state)


class MeanFieldVI:
    """Bounded optimizer; each ``step`` commits a complete Adam update atomically.

    ``init`` is a mu vector or ``{mu: [...], rho: [...]}``. With no init, means
    are independent N(0, .1) values and rho is zero. Independent PCG64 child
    streams control initialization, optimization, evaluation, and q sampling.
    Scales outside exp([-14, 8]) fail explicitly; gradients are never clipped.
    """

    def __init__(self, target: Target, dimension: int, seed: int = 42, init=None,
                 learning_rate: float = 0.03, batch_size: int = 8,
                 max_iterations: int = 1000):
        self.target = target
        self.dimension = _integer(dimension, "dimension", 1, 32)
        self.batch_size = _integer(batch_size, "batch_size", 1, 1024)
        self.max_iterations = _integer(max_iterations, "max_iterations", 1, 10000)
        self.seed = _integer(seed, "seed", 0, 2**53 - 1)
        if isinstance(learning_rate, bool):
            raise ValueError("learning_rate must be a finite number")
        self.learning_rate = float(learning_rate)
        if not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be finite and in (0, 1]")
        init_seed, opt_seed, eval_seed, sample_seed = np.random.SeedSequence(self.seed).spawn(4)
        initialization = np.random.default_rng(init_seed)
        self.rng = np.random.default_rng(opt_seed)
        self.evaluation_rng = np.random.default_rng(eval_seed)
        self.sample_rng = np.random.default_rng(sample_seed)
        if isinstance(init, dict):
            if set(init) - {"mu", "rho"}:
                raise ValueError("VI initialization accepts only mu and rho")
            self.mu = self._vector(init.get("mu", np.zeros(self.dimension)), "mu")
            self.rho = self._vector(init.get("rho", np.zeros(self.dimension)), "rho")
        else:
            self.mu = self._vector(initialization.normal(0, 0.1, self.dimension) if init is None else init, "mu")
            self.rho = np.zeros(self.dimension, dtype=np.float64)
        self._check_parameters(self.mu, self.rho)
        self.first_moment = np.zeros(2 * self.dimension, dtype=np.float64)
        self.second_moment = self.first_moment.copy()
        self.iteration = 0
        self.gradient_evaluations = 0
        self.evaluation_gradient_evaluations = 0
        self.history: list[dict] = []
        self.evaluation_history: list[dict] = []
        self.status = "ready"
        self.failure_reason: str | None = None

    def _vector(self, values, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.dimension,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {self.dimension} finite values")
        return array.copy()

    @staticmethod
    def _check_parameters(mu: np.ndarray, rho: np.ndarray):
        if not np.isfinite(mu).all() or not np.isfinite(rho).all():
            raise VIFailure("nonfinite variational parameters")
        if np.any(rho < MIN_RHO):
            raise VIFailure(f"variational scale collapse: log standard deviation below {MIN_RHO}")
        if np.any(rho > MAX_RHO):
            raise VIFailure(f"variational scale exceeded limit: log standard deviation above {MAX_RHO}")

    @property
    def scale(self) -> np.ndarray:
        return np.exp(self.rho)

    @property
    def entropy(self) -> float:
        return float(self.rho.sum() + self.dimension * 0.5 * (1 + LOG_2PI))

    def _batch(self, epsilon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = self.mu + self.scale * epsilon
        logps = np.empty(len(epsilon), dtype=np.float64)
        gradients = np.empty_like(positions)
        for index, position in enumerate(positions):
            if not np.isfinite(position).all():
                raise VIFailure("nonfinite variational sample")
            try:
                value, gradient = self.target(position)
            except (ValueError, FloatingPointError, OverflowError) as exc:
                raise VIFailure(f"target evaluation failed: {exc}") from exc
            gradient = np.asarray(gradient, dtype=np.float64)
            if not np.isfinite(value) or gradient.shape != (self.dimension,) or not np.isfinite(gradient).all():
                raise VIFailure("target returned nonfinite density or invalid gradient")
            logps[index], gradients[index] = value, gradient
        return logps, gradients

    def step(self) -> dict:
        """Apply one ascent update, recording the pre-update stochastic objective.

        Failure retains parameters, moments and optimization RNG from the last
        complete update; status and failure reason record the failed attempt.
        """
        if self.status == "failed":
            raise VIFailure(self.failure_reason or "VI has failed")
        if self.iteration >= self.max_iterations:
            self.status = "complete"
            raise StopIteration("VI iteration budget reached")
        candidate_rng = _clone_rng(self.rng)
        epsilon = candidate_rng.standard_normal((self.batch_size, self.dimension))
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                logps, gradients = self._batch(epsilon)
                gradient = np.concatenate((gradients.mean(axis=0),
                                           (gradients * epsilon * self.scale).mean(axis=0) + 1))
                if not np.isfinite(gradient).all():
                    raise VIFailure("nonfinite reparameterization gradient")
                iteration = self.iteration + 1
                first = 0.9 * self.first_moment + 0.1 * gradient
                second = 0.999 * self.second_moment + 0.001 * gradient**2
                update = self.learning_rate * (first / (1 - 0.9**iteration)) / (np.sqrt(second / (1 - 0.999**iteration)) + 1e-8)
                new_mu, new_rho = np.split(np.concatenate((self.mu, self.rho)) + update, 2)
                self._check_parameters(new_mu, new_rho)
                objective = float(logps.mean() + self.entropy)
                metadata = {"iteration": iteration, "optimization_elbo": objective,
                            "gradient_norm": float(np.linalg.norm(gradient)),
                            "min_scale": float(np.exp(new_rho).min()),
                            "max_scale": float(np.exp(new_rho).max()),
                            "gradient_evaluations": self.gradient_evaluations + self.batch_size,
                            "status": "complete" if iteration >= self.max_iterations else "running"}
                if not np.isfinite(objective) or not np.isfinite(metadata["gradient_norm"]):
                    raise VIFailure("nonfinite optimization objective or gradient norm")
        except (VIFailure, FloatingPointError, OverflowError) as exc:
            self.status, self.failure_reason = "failed", str(exc)
            raise VIFailure(self.failure_reason) from exc
        self.mu, self.rho = new_mu, new_rho
        self.first_moment, self.second_moment = first, second
        self.rng, self.iteration = candidate_rng, iteration
        self.gradient_evaluations += self.batch_size
        self.status = metadata["status"]
        self.history.append(metadata)
        return dict(metadata)

    def evaluate(self, n: int = 128) -> dict:
        """Independent ELBO estimate plus its Monte Carlo standard error.

        This is E_q[log_target_z] plus analytic entropy. It is a lower bound
        on log evidence only when target_z is a correctly normalized log joint.
        Evaluation never consumes optimization or sampling randomness.
        """
        n = _integer(n, "evaluation draws", 2, 100000)
        if len(self.evaluation_history) >= 10001:
            raise ValueError("evaluation history limit reached")
        rng = _clone_rng(self.evaluation_rng)
        values, _ = self._batch(rng.standard_normal((n, self.dimension)))
        metadata = {"iteration": self.iteration, "elbo": float(values.mean() + self.entropy),
                    "mcse": float(values.std(ddof=1) / np.sqrt(n)), "n": n}
        if not np.isfinite(metadata["elbo"]) or not np.isfinite(metadata["mcse"]):
            raise VIFailure("nonfinite objective evaluation")
        self.evaluation_rng = rng
        self.evaluation_gradient_evaluations += n
        self.evaluation_history.append(metadata)
        return dict(metadata)

    def sample(self, n: int = 500) -> np.ndarray:
        """Draw independent unconstrained q samples; transform with the model API."""
        n = _integer(n, "sample draws", 1, 8000)
        rng = _clone_rng(self.sample_rng)
        with np.errstate(over="ignore", invalid="ignore"):
            samples = self.mu + self.scale * rng.standard_normal((n, self.dimension))
        if not np.isfinite(samples).all():
            raise VIFailure("nonfinite variational sample")
        self.sample_rng = rng
        return samples

    def state_dict(self, history_start: int = 0, evaluation_start: int = 0) -> dict:
        """Versioned data only, suitable for strict JSON experiment checkpoints."""
        return {"version": 1, "algorithm": "mean-field-adam", "dimension": self.dimension,
                "seed": self.seed, "learning_rate": self.learning_rate, "batch_size": self.batch_size,
                "max_iterations": self.max_iterations, "iteration": self.iteration,
                "mu": self.mu.tolist(), "rho": self.rho.tolist(),
                "first_moment": self.first_moment.tolist(), "second_moment": self.second_moment.tolist(),
                "rng": encode_rng_state(self.rng.bit_generator.state),
                "evaluation_rng": encode_rng_state(self.evaluation_rng.bit_generator.state),
                "sample_rng": encode_rng_state(self.sample_rng.bit_generator.state),
                "gradient_evaluations": self.gradient_evaluations,
                "evaluation_gradient_evaluations": self.evaluation_gradient_evaluations,
                "history": deepcopy(self.history[history_start:]), "evaluation_history": deepcopy(self.evaluation_history[evaluation_start:]),
                "status": self.status, "failure_reason": self.failure_reason}

    def clone(self, target: Target | None = None) -> "MeanFieldVI":
        """Copy trusted in-memory state for a provisional runtime transaction.

        Importers must use ``from_state`` for validation. Internal transactions
        can avoid serializing and revalidating every historical point on every
        update. Existing history dictionaries are immutable append-only records.
        """
        other = object.__new__(type(self))
        other.__dict__ = self.__dict__.copy()
        other.target = self.target if target is None else target
        for name in ("mu", "rho", "first_moment", "second_moment"):
            setattr(other, name, getattr(self, name).copy())
        for name in ("rng", "evaluation_rng", "sample_rng"):
            setattr(other, name, _clone_rng(getattr(self, name)))
        other.history = self.history.copy()
        other.evaluation_history = self.evaluation_history.copy()
        return other

    @classmethod
    def from_state(cls, target: Target, state: dict) -> "MeanFieldVI":
        fields = {"version", "algorithm", "dimension", "seed", "learning_rate", "batch_size",
                  "max_iterations", "iteration", "mu", "rho", "first_moment", "second_moment",
                  "rng", "evaluation_rng", "sample_rng", "gradient_evaluations",
                  "evaluation_gradient_evaluations", "history", "evaluation_history", "status", "failure_reason"}
        if not isinstance(state, dict) or set(state) != fields or type(state.get("version")) is not int or state["version"] != 1 or state.get("algorithm") != "mean-field-adam":
            raise ValueError("unsupported VI checkpoint")
        dimension = _integer(state["dimension"], "dimension", 1, 32)
        _checkpoint_number(state["learning_rate"], "learning rate", 0)
        if hasattr(target, "dimension") and target.dimension != dimension:
            raise ValueError("VI checkpoint dimension differs from target")
        for name in ("mu", "rho", "first_moment", "second_moment"):
            _checkpoint_vector(state[name], dimension if name in ("mu", "rho") else 2 * dimension, name)
        obj = cls(target, state["dimension"], state["seed"], {"mu": state["mu"], "rho": state["rho"]},
                  state["learning_rate"], state["batch_size"], state["max_iterations"])
        obj.iteration = _integer(state["iteration"], "iteration", 0, obj.max_iterations)
        for name in ("first_moment", "second_moment"):
            values = np.asarray(state[name], dtype=np.float64)
            if values.shape != (2 * obj.dimension,) or not np.isfinite(values).all():
                raise ValueError(f"invalid VI {name}")
            if name == "second_moment" and np.any(values < 0):
                raise ValueError("negative Adam variance")
            setattr(obj, name, values.copy())
        for name in ("rng", "evaluation_rng", "sample_rng"):
            getattr(obj, name).bit_generator.state = _checkpoint_rng(decode_rng_state(state[name]))
        obj.gradient_evaluations = _integer(state["gradient_evaluations"], "gradient_evaluations", 0, 10240000)
        if obj.gradient_evaluations != obj.iteration * obj.batch_size:
            raise ValueError("VI gradient count differs from completed updates")
        obj.evaluation_gradient_evaluations = _integer(state["evaluation_gradient_evaluations"], "evaluation_gradient_evaluations", 0, 1000100000)
        if not isinstance(state["history"], list) or len(state["history"]) != obj.iteration:
            raise ValueError("VI optimization history does not match iteration count")
        if not isinstance(state["evaluation_history"], list) or len(state["evaluation_history"]) > 10001:
            raise ValueError("invalid VI evaluation history")
        history_keys = {"iteration", "optimization_elbo", "gradient_norm", "min_scale", "max_scale", "gradient_evaluations", "status"}
        for index, entry in enumerate(state["history"], 1):
            if not isinstance(entry, dict) or set(entry) != history_keys:
                raise ValueError("invalid VI optimization history fields")
            if _integer(entry["iteration"], "history iteration", 1, obj.max_iterations) != index:
                raise ValueError("VI history iterations are not sequential")
            if _integer(entry["gradient_evaluations"], "history evaluations", 1, 10240000) != index * obj.batch_size:
                raise ValueError("VI history gradient count mismatch")
            _checkpoint_number(entry["optimization_elbo"], "optimization ELBO")
            _checkpoint_number(entry["gradient_norm"], "gradient norm", 0)
            low = _checkpoint_number(entry["min_scale"], "minimum scale", float(np.exp(MIN_RHO)))
            high = _checkpoint_number(entry["max_scale"], "maximum scale", low)
            if high > np.exp(MAX_RHO) or entry["status"] != ("complete" if index == obj.max_iterations else "running"):
                raise ValueError("VI history has invalid scales or status")
        evaluation_count = 0
        previous_iteration = 0
        for entry in state["evaluation_history"]:
            if not isinstance(entry, dict) or set(entry) != {"iteration", "elbo", "mcse", "n"}:
                raise ValueError("invalid VI evaluation history fields")
            previous_iteration = _integer(entry["iteration"], "evaluation iteration", previous_iteration, obj.iteration)
            evaluation_count += _integer(entry["n"], "evaluation draws", 2, 100000)
            _checkpoint_number(entry["elbo"], "evaluation ELBO")
            _checkpoint_number(entry["mcse"], "evaluation MCSE", 0)
        if evaluation_count != obj.evaluation_gradient_evaluations:
            raise ValueError("VI evaluation count differs from evaluation history")
        obj.history = deepcopy(state["history"])
        obj.evaluation_history = deepcopy(state["evaluation_history"])
        if state["status"] not in {"ready", "running", "complete", "failed"}:
            raise ValueError("invalid VI status")
        reason = state["failure_reason"]
        if (state["status"] == "failed" and (not isinstance(reason, str) or not reason or len(reason) > 4096)) or (state["status"] != "failed" and reason is not None):
            raise ValueError("invalid VI failure reason")
        expected_status = "ready" if obj.iteration == 0 else "complete" if obj.iteration == obj.max_iterations else "running"
        if state["status"] not in {expected_status, "failed"}:
            raise ValueError("VI status differs from iteration count")
        obj.status, obj.failure_reason = state["status"], state["failure_reason"]
        return obj


def fit_vi(target: Target, dimension: int, iterations: int = 1000, *, seed: int = 42,
           init=None, learning_rate: float = 0.03, batch_size: int = 8,
           evaluation_interval: int = 50, evaluation_draws: int = 128) -> MeanFieldVI:
    """Bounded convenience fit; returned optimizer retains actual histories."""
    iterations = _integer(iterations, "iterations", 1, 10000)
    evaluation_interval = _integer(evaluation_interval, "evaluation_interval", 1, 10000)
    optimizer = MeanFieldVI(target, dimension, seed, init, learning_rate, batch_size, iterations)
    optimizer.evaluate(evaluation_draws)
    for iteration in range(1, iterations + 1):
        optimizer.step()
        if iteration % evaluation_interval == 0 or iteration == iterations:
            optimizer.evaluate(evaluation_draws)
    return optimizer
