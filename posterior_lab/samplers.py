"""Unit-kinetic HMC and Hoffman--Gelman (2014), Algorithm 6 slice NUTS.

All states, momenta and turning tests use the *working* coordinates. A caller
must keep its target/coordinate transformation fixed for a whole transition.
Cancellation exceptions propagate: no partial transition is a returned draw.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

Target = Callable[[np.ndarray], tuple[float, np.ndarray]]
MAX_TRAJECTORY_POINTS = 256
MAX_BRANCH_RECORDS = 128
ENERGY_THRESHOLD = 1000.0


def _position(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or not 1 <= value.size <= 32 or not np.all(np.isfinite(value)):
        raise ValueError("position must be a finite vector of dimension 1 to 32")
    return value.copy()


def _evaluate(target, q):
    """Convert numerical target failures to an invalid proposal, never clipping."""
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            lp, grad = target(q)
        lp, grad = float(lp), np.asarray(grad, dtype=np.float64)
        if grad.shape != q.shape:
            raise ValueError("target gradient shape differs from position")
        if not np.isfinite(lp) or not np.all(np.isfinite(grad)):
            return -math.inf, np.full_like(q, np.nan)
        return lp, grad
    except (FloatingPointError, OverflowError):
        return -math.inf, np.full_like(q, np.nan)


def leapfrog(target: Target, position, momentum, gradient, step_size):
    """One reversible kick--drift--kick step; gradient means gradient of log p.

    The signed step permits backward integration. Inputs are never mutated.
    Invalid target evaluations return logp=-inf and nonfinite gradients.
    """
    q = np.asarray(position, dtype=np.float64)
    r = np.asarray(momentum, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    if q.ndim != 1 or r.shape != q.shape or g.shape != q.shape:
        raise ValueError("position, momentum and gradient must be matching vectors")
    if not np.isfinite(step_size) or step_size == 0:
        raise ValueError("step size must be finite and nonzero")
    with np.errstate(over="ignore", invalid="ignore"):
        half = r + 0.5 * float(step_size) * g
        next_q = q + float(step_size) * half
        # Call even for a nonfinite q so the caller's gradient accounting and
        # cancellation hook remain exact. The target wrapper rejects it.
        lp, next_g = _evaluate(target, next_q)
        next_r = half + 0.5 * float(step_size) * next_g
    return next_q, next_r, lp, next_g


def _joint(lp, r):
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(lp - 0.5 * np.dot(r, r))
    return value if np.isfinite(value) else -math.inf


def _acceptance(log_ratio):
    return float(math.exp(min(0.0, log_ratio))) if np.isfinite(log_ratio) else 0.0


def _log_uniform(rng):
    # random() can equal zero. -inf is the exact log of that endpoint.
    u = float(rng.random())
    return math.log(u) if u > 0 else -math.inf


def _finite_number(value):
    return float(value) if np.isfinite(value) else None


def _safe_list(value):
    return [_finite_number(v) for v in value]


class _Context:
    def __init__(self, target, check, record, initial_joint=0.0):
        self.target, self.check, self.record = target, check, bool(record)
        self.initial_joint = initial_joint
        self.evaluations = self.steps = 0
        self.trajectory, self.branches = [], []
        self.omitted_points = self.omitted_branches = 0
        self.max_energy_error = 0.0
        self.divergent = False

    def evaluate(self, q):
        if self.check is not None:
            self.check()  # Deliberately outside numerical failure handlers.
        self.evaluations += 1
        if not np.all(np.isfinite(q)):
            return -math.inf, np.full_like(q, np.nan)
        return _evaluate(self.target, q)

    def point(self, state, **extra):
        if not self.record:
            return
        if len(self.trajectory) >= MAX_TRAJECTORY_POINTS:
            self.omitted_points += 1
            return
        self.trajectory.append({"position": _safe_list(state.q),
                                "momentum": _safe_list(state.r),
                                "log_density": _finite_number(state.lp),
                                "integration_step": self.steps, **extra})

    def branch(self, **extra):
        if not self.record:
            return
        if len(self.branches) >= MAX_BRANCH_RECORDS:
            self.omitted_branches += 1
            if extra.get("kind") == "expansion":
                # Retain all (at most nine) top-level decisions even when a
                # deep tree exhausts the recursive-branch visualization budget.
                remove = next((i for i, b in enumerate(self.branches)
                               if b.get("kind") == "recursive"), None)
                if remove is not None:
                    self.branches.pop(remove)
                    self.branches.append(extra)
        else:
            self.branches.append(extra)


@dataclass
class _State:
    q: np.ndarray
    r: np.ndarray
    lp: float
    g: np.ndarray


def _start(target, position, rng, step_size, check, record):
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step size must be finite and positive")
    q = _position(position)
    context = _Context(target, check, record)
    lp, g = context.evaluate(q)
    if not np.isfinite(lp):
        raise ValueError("initial position has nonfinite log density or gradient")
    r = np.asarray(rng.normal(size=q.size), dtype=np.float64)
    state = _State(q, r, lp, g)
    context.initial_joint = _joint(lp, r)
    context.point(state, kind="initial", eligible=True)
    return state, context


def _advance(state, step, context):
    context.steps += 1
    return _State(*leapfrog(context.evaluate, state.q, state.r, state.g, step))


def _metadata(context, start, proposed, selected, **extra):
    return {
        "coordinates": "scaled working x; z = scales * x",
        "starting_position": start.q.tolist(),
        "starting_log_density": start.lp,
        "initial_momentum": start.r.tolist(),
        "proposed_position": _safe_list(proposed.q),
        "selected_position": selected.q.tolist(),
        "selected_log_density": selected.lp,
        "selected_gradient": selected.g.tolist(),
        "gradient_evaluations": context.evaluations,
        "integration_steps": context.steps,
        "divergent": bool(context.divergent),
        "max_abs_energy_error": _finite_number(context.max_energy_error),
        "trajectory": context.trajectory,
        "branches": context.branches,
        "omitted_trajectory_points": context.omitted_points,
        "omitted_branches": context.omitted_branches,
        **extra,
    }


def hmc_transition(target: Target, position, rng, step_size, steps=10,
                   record=False, check=None):
    """Complete HMC transition, including repeated position on rejection."""
    if isinstance(steps, bool) or int(steps) != steps or not 1 <= steps <= 4096:
        raise ValueError("HMC integration steps must be an integer from 1 to 4096")
    start, context = _start(target, position, rng, step_size, check, record)
    state = start
    error = 0.0
    for _ in range(int(steps)):
        state = _advance(state, step_size, context)
        error = context.initial_joint - _joint(state.lp, state.r)
        context.max_energy_error = max(context.max_energy_error, abs(error))
        invalid = not np.isfinite(error)
        context.divergent |= invalid or abs(error) > ENERGY_THRESHOLD
        context.point(state, kind="leapfrog", energy_error=_finite_number(error),
                      eligible=not invalid)
        if invalid:
            break
    probability = _acceptance(-error)
    # The momentum flip makes the deterministic proposal an involution.
    proposed = _State(state.q, -state.r, state.lp, state.g)
    accepted = bool(_log_uniform(rng) < min(0.0, -error))
    selected = proposed if accepted else start
    metadata = _metadata(
        context, start, proposed, selected, algorithm="hmc", acceptance=probability,
        step_size=float(step_size), requested_steps=int(steps),
        accepted=accepted, energy_error=_finite_number(error),
        proposed_momentum=_safe_list(proposed.r), depth=0,
        max_depth_reached=False, stop_reason="completed" if np.isfinite(error) else "nonfinite proposal",
    )
    return selected.q.copy(), metadata


def _turning(minus, plus):
    displacement = plus.q - minus.q
    with np.errstate(over="ignore", invalid="ignore"):
        return bool(np.dot(displacement, minus.r) < 0 or
                    np.dot(displacement, plus.r) < 0 or
                    not np.all(np.isfinite(displacement)))


@dataclass
class _Tree:
    minus: _State
    plus: _State
    candidate: _State
    count: int
    continuing: bool
    alpha: float
    n_alpha: int


def _recursive_probability(left_count, right_count):
    total = left_count + right_count
    return right_count / total if total else 0.0


def _top_level_probability(old_count, new_count):
    return min(1.0, new_count / old_count)


def _build_tree(state, log_slice, direction, depth, step_size, rng, context):
    if depth == 0:
        leaf = _advance(state, direction * step_size, context)
        joint = _joint(leaf.lp, leaf.r)
        energy_error = context.initial_joint - joint
        finite = bool(np.isfinite(joint))
        count = int(finite and log_slice <= joint)
        # Published Algorithm 6 condition is slice-relative. Do not replace it
        # by energy_error < threshold, which is a different stopping rule.
        continuing = bool(finite and log_slice < joint + ENERGY_THRESHOLD)
        context.divergent |= not continuing
        context.max_energy_error = max(context.max_energy_error, abs(energy_error))
        context.point(leaf, kind="leapfrog", direction=direction,
                      slice_eligible=bool(count), eligible=bool(count),
                      continuing=continuing, energy_error=_finite_number(energy_error))
        return _Tree(leaf, leaf, leaf, count, continuing,
                     _acceptance(joint - context.initial_joint), 1)

    first = _build_tree(state, log_slice, direction, depth - 1, step_size, rng, context)
    if not first.continuing:
        context.branch(kind="recursive", depth=depth, direction=direction,
                       continuing=False, stop_reason="first subtree stopped",
                       candidate_count=first.count)
        return first
    edge = first.minus if direction == -1 else first.plus
    second = _build_tree(edge, log_slice, direction, depth - 1, step_size, rng, context)
    minus = second.minus if direction == -1 else first.minus
    plus = first.plus if direction == -1 else second.plus
    probability = _recursive_probability(first.count, second.count)
    choose_second = float(rng.random()) < probability
    candidate = second.candidate if choose_second else first.candidate
    turns = _turning(minus, plus)
    continuing = second.continuing and not turns
    context.branch(kind="recursive", depth=depth, direction=direction,
                   first_count=first.count, second_count=second.count,
                   selection_probability=probability, selected_second=choose_second,
                   continuing=continuing, internal_uturn=turns,
                   stop_reason="subtree stopped" if not second.continuing else
                               "internal U-turn" if turns else "continue")
    return _Tree(minus, plus, candidate, first.count + second.count,
                 continuing, first.alpha + second.alpha,
                 first.n_alpha + second.n_alpha)


def nuts_transition(target: Target, position, rng, step_size, max_depth=7,
                    record=False, check=None):
    """Original slice-based NUTS, Algorithm 6, with bounded maximum depth.

    ``acceptance`` is alpha/n_alpha from the *last doubling subtree*, exactly
    as in the paper. It is the dual-averaging statistic, not a boolean move or
    a final Metropolis probability. No multinomial or momentum-sum rules.
    """
    if isinstance(max_depth, bool) or int(max_depth) != max_depth or not 1 <= max_depth <= 9:
        raise ValueError("NUTS maximum depth must be an integer from 1 to 9")
    start, context = _start(target, position, rng, step_size, check, record)
    log_slice = context.initial_joint + _log_uniform(rng)
    minus = plus = selected = proposed = start
    count, depth, continuing = 1, 0, True
    alpha, n_alpha, all_alpha, all_n_alpha = 0.0, 1, 0.0, 0
    stop_reason = "maximum depth"
    while continuing and depth < int(max_depth):
        direction = -1 if float(rng.random()) < 0.5 else 1
        edge = minus if direction == -1 else plus
        first_step = context.steps + 1
        tree = _build_tree(edge, log_slice, direction, depth, step_size, rng, context)
        if record and not tree.continuing:
            # A leaf can lie inside the slice while belonging to a subtree
            # that is excluded in its entirety. Preserve both facts for replay.
            for point in context.trajectory:
                if point["integration_step"] >= first_step:
                    point["eligible"] = False
                    point["exclusion_reason"] = "new subtree stopped before selection"
        if direction == -1:
            minus = tree.minus
        else:
            plus = tree.plus
        proposed = tree.candidate
        probability = _top_level_probability(count, tree.count)
        choose_new = bool(tree.continuing and float(rng.random()) < probability)
        if choose_new:
            selected = tree.candidate
        turns = _turning(minus, plus)
        continuing = tree.continuing and not turns
        context.branch(kind="expansion", depth=depth, direction=direction,
                       first_integration_step=first_step, last_integration_step=context.steps,
                       old_count=count, new_count=tree.count,
                       selection_probability=probability, selected_new=choose_new,
                       subtree_valid=tree.continuing, global_uturn=turns,
                       continuing=continuing)
        count += tree.count
        depth += 1
        # These are intentionally replaced (not accumulated) for Algorithm 6.
        alpha, n_alpha = tree.alpha, tree.n_alpha
        all_alpha += tree.alpha
        all_n_alpha += tree.n_alpha
        if not tree.continuing:
            stop_reason = "divergence" if context.divergent else "internal U-turn"
        elif turns:
            stop_reason = "U-turn"
    error = context.initial_joint - _joint(selected.lp, selected.r)
    return selected.q.copy(), _metadata(
        context, start, proposed, selected, algorithm="nuts_slice_algorithm6",
        step_size=float(step_size), requested_max_depth=int(max_depth),
        acceptance=alpha / n_alpha,
        acceptance_all_leaves=all_alpha / all_n_alpha,
        acceptance_definition="mean acceptance in final doubling subtree (Algorithm 6)",
        accepted=bool(not np.array_equal(start.q, selected.q)),
        energy_error=_finite_number(error), depth=depth,
        max_depth_reached=bool(continuing and depth == max_depth),
        candidate_count=count, log_slice=_finite_number(log_slice), stop_reason=stop_reason,
    )
