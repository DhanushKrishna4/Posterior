# Sampling coordinates and adaptation

Posterior uses float64 arrays and unit kinetic energy in **working coordinates**:

```
z = scales * x
log_target_x(x) = log_target_z(scales * x) + sum(log(scales))
gradient_x = scales * gradient_z
H(x, r) = -log_target_x(x) + dot(r, r) / 2
```

The model engine has already included parameter-transform Jacobians in
`log_target_z`. The coordinate determinant above is additional because it is a
different change of variables. Scale estimation receives unconstrained `z`,
not constrained parameters. Every transition captures fixed scales. At a scale
window boundary, retain the same `z` and recompute `x = z / scales`.

## HMC

The integrator uses half a momentum update with **positive log-density
gradient**, a full position update, and another half momentum update. Signed
steps enable backward integration. HMC refreshes standard-normal momentum,
integrates a configurable number of steps, flips the proposal momentum, and
performs the Metropolis comparison in log space. A rejection returns the prior
position as a completed, repeated draw. Nonfinite proposals reject explicitly;
neither parameters nor gradients are clipped. See [Neal's HMC treatment,
Figure 2](https://arxiv.org/abs/1206.1901).

## Exact NUTS variant

This is the **original slice-based NUTS, Hoffman and Gelman (2014), Algorithm
6**, read directly from [the paper, page 1611](https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf).
It uses a log slice, random signed expansion, recursive subtrees and endpoint
momentum turning tests in working coordinates. It does not use multinomial
selection or momentum-sum termination.

Inside recursion, choose the second subtree candidate with probability
`n_second / (n_first + n_second)`. At top-level expansion, the probability is
`min(1, n_new / n_old)`, conditional on the new subtree remaining valid. These
rules are intentionally different. Zero-count recursion chooses neither new
candidate. Internal U-turns invalidate the entire new subtree for top-level
selection.

The continuation test is `log_slice < joint_at_leaf + 1000`, distinct from an
energy-error test. Metadata reports slice-condition failure or nonfinite
arithmetic as divergence, and records energy error separately. The adaptation
acceptance statistic averages the **last doubling subtree**, as in Algorithm
6; `acceptance_all_leaves` is additional inspection data. A depth limit ends
expansion separately and is not divergence.

## Windowed warmup

The architecture follows the [three-stage structure described by
Stan](https://mc-stan.org/docs/reference-manual/mcmc.html), without claiming
implementation equivalence. For at least 50 warmup draws, initial adaptation
uses `max(10, floor(.15 * warmup))` iterations, final adaptation uses
`max(10, floor(.10 * warmup))`, and the middle interval uses growing windows.
The initial window has at most 25 observations; successive widths double,
merging the last remainder when another complete doubled window cannot fit.
Fewer than 50 warmup draws use identity scales and step-size adaptation only.

Each window has fresh Welford accumulators. Its variance estimate is
`n/(n+5) * sample_variance + 5/(n+5) * .001`, bounded to `[.0001, 10000]`.
Scales are its square roots. At every window end, discard that accumulator and
restart dual averaging from its smoothed step. This regularization and these
bounds are Posterior choices, not guarantees of adequate exploration.

Dual averaging uses target acceptance (default .8), gamma .05, t0 10 and
kappa .75. Steps are bounded to `[.00001, 4]`. Initial step-size search takes
at most 20 trial steps by default. At warmup completion, freeze smoothed step
size and scales. Every chain owns separate state, including both accumulators.

## Records, cancellation and limitations

Returned metadata contains start/proposal/selected states, initial momentum,
energy error, actual target evaluation counts, integration steps, acceptance
statistics, depth and stop reason. Detailed records are opt-in and hold at most
256 trajectory points and 128 branch records, with omitted counts. They cannot
influence the RNG or transition. Intermediate points are **not posterior
draws**, and all recorded positions are working `x` coordinates.

The optional `check()` hook runs before every gradient evaluation. Cancellation
or budget exceptions escape without a returned transition. The transaction
owner must restore its RNG and avoid committing partial adaptation or draws.
The sampler itself never mutates the supplied position. Production NUTS depth
is capped at 9 (at most 511 leapfrog steps), HMC at 4096 steps, and parameter
dimension at 32.

Diagonal scaling cannot remove correlations, nonlinear funnel curvature, or
barriers between separated modes. High acceptance does not establish useful
exploration. Fixed HMC trajectory lengths may resonate with a target. Inspect
retained-chain diagnostics and actual geometry rather than interpreting a
completed run as proof of accuracy.

## Reproducible native validation

Run `.venv/bin/python scripts/validate_samplers.py` to reproduce
[`reports/sampler-validation.json`](../reports/sampler-validation.json). It
contains all retained model and unconstrained draws, initial positions, fixed
seeds, final RNG states, source hashes, environment, numerical errors and work
counts. A measured 50-transition pilot precedes the six full cases. All run
sequentially. The case order is Beta–Binomial, correlated Gaussian and conjugate
regression, each with HMC and NUTS.

The checked run retained 750 draws for each of four seeds (17, 29, 53, 101)
after 100 burn-in draws. It used fixed step .2, HMC length 5 and NUTS depth 7.
Regression uses oracle marginal standard deviations as fixed coordinates;
this report validates the kernels and graph together, while windowed warmup
has separate tests. All six cases passed the declared gates: R-hat below 1.01,
bulk ESS above 400, mean error below five raw-mean MCSEs, standardized covariance
error below .2 and quantile error below .25 marginal SDs. No case diverged.
Native case durations were 2.01–7.72 seconds on the recorded arm64 environment;
these are measurements rather than browser runtime predictions.

The report also preserves an actual failure from HMC with Beta–Binomial step
.35 and length 10: estimated probability mean .60633 versus exact .64286,
R-hat 1.168 and bulk ESS 14.2. Its integration time is close to the target's
local oscillation period. That failed mean check was investigated; seeds and
tolerances were not changed to rescue it. The failure is an explicit artifact,
not a scripted visual effect.
