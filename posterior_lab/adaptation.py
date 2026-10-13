"""Dual averaging and windowed diagonal *coordinate* adaptation.

The transformation is z = scales * x, not an implicit mass convention. Scale
estimates use unconstrained model-space draws and are fixed per transition.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .samplers import _evaluate, _joint, _acceptance, leapfrog

MIN_STEP = 1e-5
MAX_STEP = 4.0
MIN_VARIANCE = 1e-4
MAX_VARIANCE = 1e4


def _finite_scalar(value, name, lower=-math.inf, upper=math.inf):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not lower <= value <= upper):
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return float(value)


def _bounded_integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _state_vector(value, name, dimension=None, lower=-math.inf, upper=math.inf):
    # Validate the JSON container and its scalar leaves before NumPy allocates.
    if (not isinstance(value, list) or not 1 <= len(value) <= 32
            or (dimension is not None and len(value) != dimension)):
        raise ValueError(f"{name} must be a vector matching dimension 1 to 32")
    for item in value:
        _finite_scalar(item, name, lower, upper)
    return np.asarray(value, dtype=np.float64)


def coordinate_target(target_z, scales):
    """Capture a fixed diagonal S and return log p_z(Sx) + log|S|, S grad_z."""
    scales = np.asarray(scales, dtype=np.float64).copy()
    if scales.ndim != 1 or not 1 <= scales.size <= 32 or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("coordinate scales must be a finite positive vector")
    log_det = float(np.log(scales).sum())

    def evaluate(x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != scales.shape:
            raise ValueError("working position and scales have different shapes")
        lp, grad = target_z(scales * x)
        return float(lp) + log_det, scales * np.asarray(grad, dtype=np.float64)

    return evaluate


def find_reasonable_step_size(target, position, rng, initial=0.1, max_trials=20,
                              check=None, return_info=False):
    """Bounded Algorithm 4 style search, testing the same refreshed momentum.

    At most max_trials trial gradients plus one initial gradient are evaluated.
    The returned crossing step is constrained to [1e-5, 4]. Hitting a bound is
    reported in optional metadata, never represented as successful adaptation.
    """
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError("initial step size must be finite and positive")
    if isinstance(max_trials, bool) or int(max_trials) != max_trials or not 1 <= max_trials <= 30:
        raise ValueError("step search requires between 1 and 30 trials")
    q = np.asarray(position, dtype=np.float64).copy()
    evaluations = 0

    def evaluate(x):
        nonlocal evaluations
        if check is not None:
            check()
        evaluations += 1
        if not np.all(np.isfinite(x)):
            return -math.inf, np.full_like(x, np.nan)
        return _evaluate(target, x)

    lp, grad = evaluate(q)
    if not np.isfinite(lp):
        raise ValueError("cannot search step size at an invalid initial position")
    momentum = np.asarray(rng.normal(size=q.size), dtype=np.float64)
    initial_joint = _joint(lp, momentum)
    step = float(np.clip(initial, MIN_STEP, MAX_STEP))
    direction = None
    status = "trial limit"
    acceptance = 0.0
    for _ in range(int(max_trials)):
        _, r, next_lp, _ = leapfrog(evaluate, q, momentum, grad, step)
        acceptance = _acceptance(_joint(next_lp, r) - initial_joint)
        if direction is None:
            direction = 1 if acceptance > 0.5 else -1
        elif (direction == 1 and acceptance <= 0.5) or (direction == -1 and acceptance >= 0.5):
            status = "crossed 0.5 acceptance"
            break
        proposed_step = float(np.clip(step * (2.0 if direction == 1 else 0.5), MIN_STEP, MAX_STEP))
        if proposed_step == step:
            status = "step size bound"
            break
        if evaluations - 1 < max_trials:
            step = proposed_step
    info = {"status": status, "gradient_evaluations": evaluations,
            "trials": evaluations - 1, "acceptance": acceptance, "step_size": step}
    return (step, info) if return_info else step


@dataclass
class DualAveraging:
    initial: float
    target: float = 0.8
    iteration: int = 0
    hbar: float = 0.0
    log_step_bar: float = 0.0
    log_step: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.initial) or not MIN_STEP <= self.initial <= MAX_STEP:
            raise ValueError("initial dual-averaging step is outside [1e-5, 4]")
        if not np.isfinite(self.target) or not 0 < self.target < 1:
            raise ValueError("target acceptance must lie strictly between zero and one")
        if self.iteration == 0:
            self.log_step = self.log_step_bar = math.log(self.initial)

    def update(self, acceptance):
        if not np.isfinite(acceptance) or not 0 <= acceptance <= 1:
            raise ValueError("acceptance statistic must be finite and in [0, 1]")
        self.iteration += 1
        t = self.iteration
        eta = 1.0 / (t + 10.0)
        self.hbar = (1 - eta) * self.hbar + eta * (self.target - acceptance)
        raw = math.log(10 * self.initial) - math.sqrt(t) / 0.05 * self.hbar
        # Bounds are on numerical tuning, never on model parameters/gradients.
        self.log_step = float(np.clip(raw, math.log(MIN_STEP), math.log(MAX_STEP)))
        weight = t ** -0.75
        self.log_step_bar = weight * self.log_step + (1 - weight) * self.log_step_bar
        return float(np.clip(math.exp(self.log_step), MIN_STEP, MAX_STEP))

    @property
    def smoothed(self):
        return float(np.clip(math.exp(self.log_step_bar), MIN_STEP, MAX_STEP))

    @classmethod
    def from_state(cls, state):
        expected = {"initial", "target", "iteration", "hbar", "log_step_bar", "log_step"}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("invalid dual-averaging checkpoint fields")
        initial = _finite_scalar(state["initial"], "dual initial step", MIN_STEP, MAX_STEP)
        target = _finite_scalar(state["target"], "dual target", 0, 1)
        if target in (0, 1):
            raise ValueError("dual target must lie strictly between zero and one")
        iteration = _bounded_integer(state["iteration"], "dual iteration", 0, 5000)
        hbar = _finite_scalar(state["hbar"], "dual hbar", -1, 1)
        lower, upper = math.log(MIN_STEP) - 1e-12, math.log(MAX_STEP) + 1e-12
        log_step = _finite_scalar(state["log_step"], "dual log step", lower, upper)
        log_step_bar = _finite_scalar(state["log_step_bar"], "dual smoothed log step", lower, upper)
        if iteration == 0 and (hbar != 0 or
                not math.isclose(log_step, math.log(initial), rel_tol=0, abs_tol=1e-12) or
                not math.isclose(log_step_bar, math.log(initial), rel_tol=0, abs_tol=1e-12)):
            raise ValueError("inconsistent restarted dual-averaging state")
        return cls(initial=initial, target=target, iteration=iteration, hbar=hbar,
                   log_step_bar=log_step_bar, log_step=log_step)


class WelfordVariance:
    def __init__(self, dim):
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != self.mean.shape or not np.all(np.isfinite(value)):
            raise ValueError("scale observation must be a finite vector matching dimension")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def regularized_variance(self):
        if self.count < 2:
            return np.ones_like(self.mean)
        sample_variance = self.m2 / (self.count - 1)
        weight = self.count / (self.count + 5.0)
        return np.clip(weight * sample_variance + (1 - weight) * 1e-3,
                       MIN_VARIANCE, MAX_VARIANCE)

    def state_dict(self):
        return {"count": self.count, "mean": self.mean.tolist(), "m2": self.m2.tolist()}

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict) or set(state) != {"count", "mean", "m2"}:
            raise ValueError("invalid Welford checkpoint fields")
        count = _bounded_integer(state["count"], "Welford count", 0, 5000)
        mean = _state_vector(state["mean"], "Welford mean")
        m2 = _state_vector(state["m2"], "Welford m2", dimension=len(mean), lower=0)
        if (count == 0 and (np.any(mean != 0) or np.any(m2 != 0))) or (count == 1 and np.any(m2 != 0)):
            raise ValueError("inconsistent empty/single-observation Welford state")
        result = cls(len(mean))
        result.count, result.mean, result.m2 = count, mean, m2
        return result


def _schedule(warmup):
    if warmup < 50:
        return warmup, warmup, []
    initial = max(10, int(0.15 * warmup))
    final = max(10, int(0.10 * warmup))
    end = warmup - final
    cursor, width, boundaries = initial, min(25, end - initial), []
    while cursor < end:
        # Merge the remainder if the next doubled window would not fit.
        if end - cursor < 3 * width:
            cursor = end
        else:
            cursor += width
        boundaries.append(cursor)
        width *= 2
    return initial, end, boundaries


class WarmupAdapter:
    """Three-stage adaptation; ``update`` is called once after a transition.

    True means scale estimates changed at this completed transition's boundary.
    The caller preserves its z and rebuilds x=z/scales for the next transition.
    Fewer than 50 warmup draws adapt step size only and keep identity scaling.
    """

    def __init__(self, dim, warmup, step_size=0.1, target_accept=0.8):
        if isinstance(dim, bool) or int(dim) != dim or not 1 <= dim <= 32:
            raise ValueError("adaptation dimension must be between 1 and 32")
        if isinstance(warmup, bool) or int(warmup) != warmup or not 0 <= warmup <= 5000:
            raise ValueError("warmup must be an integer from 0 to 5000")
        if not np.isfinite(step_size) or not MIN_STEP <= step_size <= MAX_STEP:
            raise ValueError("step size must be in [1e-5, 4]")
        self.dim, self.warmup = int(dim), int(warmup)
        self.target_accept = float(target_accept)
        self.iteration = 0
        self.scales = np.ones(self.dim, dtype=np.float64)
        self._step_size = float(step_size)
        self.dual = DualAveraging(self._step_size, self.target_accept)
        self.variance = WelfordVariance(self.dim)
        self.initial_end, self.slow_end, self.window_ends = _schedule(self.warmup)
        self.status = ("no warmup; identity scales" if warmup == 0 else
                       "short warmup; identity scales" if warmup < 50 else
                       "three-stage windowed adaptation")
        self.history = []

    @property
    def step_size(self):
        return self._step_size

    @property
    def frozen(self):
        return self.iteration >= self.warmup

    @property
    def stage(self):
        if self.frozen:
            return "frozen"
        if self.iteration < self.initial_end:
            return "initial step-size adaptation"
        if self.iteration < self.slow_end:
            return "windowed scale and step-size adaptation"
        return "final step-size adaptation"

    def update(self, z, acceptance):
        if self.frozen:
            return False
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (self.dim,) or not np.all(np.isfinite(z)):
            raise ValueError("warmup position must match dimension and be finite")
        self.iteration += 1
        self._step_size = self.dual.update(float(acceptance))
        if self.initial_end < self.iteration <= self.slow_end:
            self.variance.update(z)
        boundary = self.iteration in self.window_ends
        if boundary:
            old_scales = self.scales.copy()
            self.scales = np.sqrt(self.variance.regularized_variance())
            self.history.append({"iteration": self.iteration, "event": "scale window",
                                 "observations": self.variance.count,
                                 "old_scales": old_scales.tolist(),
                                 "scales": self.scales.tolist(),
                                 "step_size": self.dual.smoothed})
            self.variance = WelfordVariance(self.dim)
            self._step_size = self.dual.smoothed
            self.dual = DualAveraging(self._step_size, self.target_accept)
        if self.frozen:
            self._step_size = self.dual.smoothed
            self.history.append({"iteration": self.iteration, "event": "frozen",
                                 "scales": self.scales.tolist(), "step_size": self.step_size})
        return boundary

    def state_dict(self):
        return {"version": 1, "dim": self.dim, "warmup": self.warmup,
                "target_accept": self.target_accept, "iteration": self.iteration,
                "scales": self.scales.tolist(), "step_size": self.step_size,
                "dual": asdict(self.dual), "variance": self.variance.state_dict(),
                "history": self.history.copy(), "status": self.status}

    @classmethod
    def from_state(cls, state):
        fields = {"version", "dim", "warmup", "target_accept", "iteration", "scales",
                  "step_size", "dual", "variance", "history", "status"}
        if not isinstance(state, dict) or set(state) != fields:
            raise ValueError("invalid adaptation checkpoint fields")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("unsupported adaptation checkpoint version")
        dim = _bounded_integer(state["dim"], "adaptation dimension", 1, 32)
        warmup = _bounded_integer(state["warmup"], "warmup", 0, 5000)
        iteration = _bounded_integer(state["iteration"], "adaptation iteration", 0, warmup)
        scales = _state_vector(state["scales"], "coordinate scales", dim,
                               math.sqrt(MIN_VARIANCE), math.sqrt(MAX_VARIANCE))
        _finite_scalar(state["step_size"], "adaptation step size", MIN_STEP, MAX_STEP)
        _finite_scalar(state["target_accept"], "target acceptance", 0, 1)
        if not isinstance(state["history"], list) or len(state["history"]) > 5005:
            raise ValueError("invalid adaptation history size")
        if not isinstance(state["status"], str) or len(state["status"]) > 100:
            raise ValueError("invalid adaptation status")
        result = cls(state["dim"], state["warmup"], state["step_size"], state["target_accept"])
        result.iteration, result.scales = iteration, scales
        result.dual = DualAveraging.from_state(state["dual"])
        result.variance = WelfordVariance.from_state(state["variance"])
        if result.variance.mean.shape != (result.dim,) or result.variance.m2.shape != (result.dim,):
            raise ValueError("invalid variance accumulator dimension")
        previous_boundary = max([0] + [b for b in result.window_ends if b <= iteration])
        expected_dual_count = iteration - previous_boundary
        expected_variance_count = (iteration - max(result.initial_end, previous_boundary)
                                   if result.initial_end < iteration <= result.slow_end else 0)
        if result.dual.iteration != expected_dual_count or result.variance.count != expected_variance_count:
            raise ValueError("adaptation accumulator counts disagree with warmup schedule")
        if result.dual.target != result.target_accept:
            raise ValueError("dual target disagrees with target acceptance")
        expected_step = result.dual.smoothed if result.frozen else math.exp(result.dual.log_step)
        if not math.isclose(result.step_size, expected_step, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("step size disagrees with dual-averaging state")
        if previous_boundary == 0 and not np.array_equal(scales, np.ones(dim)):
            raise ValueError("scales must be identity before the first window boundary")
        if state["status"] != result.status:
            raise ValueError("adaptation status disagrees with warmup schedule")
        result.history = list(state["history"])
        return result
