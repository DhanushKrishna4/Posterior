# Diagnostics and variational approximation

All calculations use float64 and retained draws in `[chain, draw, parameter]`
order. Rejected transitions remain repeated draws. Warmup and variational draws
must not be passed into MCMC diagnostics. Plot downsampling does not affect them.

## Definitions and reference contract

Posterior implements the definitions in
[Vehtari et al. (2021)](https://arxiv.org/abs/1903.08008).
The independent development reference is **ArviZ 0.21.0**, using the public
`rhat`, `ess` and `mcse` functions; it is never a production dependency.
Tests use independent draws, AR(1) draws, shifted chains, unequal scales, ties,
heavy tails, odd draw counts, and constant chains. Diagnostic fixture seeds and
numerical tolerances are fixed in `tests/test_diagnostics.py`.

1. Split each chain into its first and last `floor(N/2)` observations. With odd
   N, discard the middle observation **only from split diagnostics**, retaining
   it in summaries and exports. Pooled quantile thresholds use all raw draws.
2. Average pooled ranks for ties. Convert rank r among S split observations to
   `Phi^-1((r - 3/8)/(S + 1/4))`.
3. For M split chains of length n, let W be the mean unbiased within-chain
   variance and B be n times the unbiased variance of chain means. R-hat is
   `sqrt(((n-1)/n * W + B/n) / W)`.
4. Apply that calculation to rank scores and separately to rank scores of
   absolute deviations from the pooled **split** median. Report both and their
   maximum. This is the rank-normalized split/folded R-hat recommended by the
   [posterior R-hat reference](https://mc-stan.org/posterior/reference/rhat.html).
5. Estimate autocovariances by zero-padded FFT with denominator n. Combine them
   using between-chain and within-chain variance, then use Geyer's initial
   positive and monotone pair sequence. The pinned reference's final positive
   even-lag correction and `tau >= 1/log10(M*n)` convention are retained. ESS is
   `M*n/tau`, and can exceed the retained count for anticorrelated sequences.
6. Bulk ESS uses split rank scores. Tail ESS is the minimum of split binary
   indicator ESS for raw observations at or below the pooled 5th and 95th
   percentiles. Mean-specific ESS uses raw split observations.
7. Mean MCSE is the raw pooled sample standard deviation divided by the square
   root of **mean-specific raw-scale ESS**, never bulk rank ESS. It estimates
   simulation error in a mean, not posterior uncertainty.

The full diagnostic definitions are also exported in each JSON report. Tests
compare seven metrics per nondegenerate fixture at relative/absolute tolerance
`2e-10`; comparisons use public standalone diagnostic calls, not ArviZ's summary
table, which has a different folded-median order in that release for odd N.

### Invalid data and interpretation

At least two chains and four draws per chain are necessary to calculate this
report; that is a mathematical minimum, not enough evidence for reliable use.
Fewer than 50 retained draws or ESS below 100 times the number of original
chains receives preliminary status. R-hat above 1.01 reports “chains disagree”.
An assessed result says only “no disagreement detected; unseen modes remain
possible”; there is no universal converged badge.

Nonfinite observations or any constant original chain make diagnostics
unavailable. Values are JSON `null`, with an explicit reason. This deliberate
safety exception follows posterior's constant-chain guidance: ArviZ 0.21.0
returns the sample count as ESS for an entirely constant variable, which would
be misleading here. A constant transformed tail indicator also has unavailable
tail ESS and makes the report unreliable. Descriptive summaries of finite
constant samples can still be shown; they do not establish sampling quality.

Inspect traces and per-chain rank histograms alongside these numbers. High
acceptance can accompany tiny steps and poor exploration. Maximum-depth events
indicate truncated tree growth and are separate from numerical divergences.
Good R-hat cannot rule out a mode missed by every chain, and equal pooling of
chains trapped in distinct modes does not recover their target probabilities.
See [Stan's diagnostic guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html).

## Mean-field VI

Following the reparameterization and analytic entropy construction in
[Kucukelbir et al. (2017), sections 2.4–2.6](https://jmlr.org/papers/v18/16-107.html),
Posterior uses `z = mu + exp(rho)*epsilon`, `epsilon ~ N(0,I)` in unconstrained
model coordinates. Its objective is
`E[log_joint(T(z), data) + log|J_T(z)|] + sum(rho) + d/2*(1+log(2*pi))`.
The target already contains each model Jacobian once. VI adds only Gaussian
entropy; deterministic model expressions receive no extra transform Jacobian.

With g the target gradient, stochastic objective gradients are
`gradient_mu = mean(g)` and `gradient_rho = mean(g*exp(rho)*epsilon) + 1`.
An implemented Adam ascent optimizer uses beta1=0.9, beta2=0.999, bias correction
and epsilon=1e-8. It never clips gradients or silently repairs scales.
The default budget is 1,000 updates with eight fresh samples per update.
Batch size is capped at 1,024, dimension at 32, and updates at 10,000.
Scales outside `[exp(-14), exp(8)]`, nonfinite targets/gradients, or arithmetic
overflow fail explicitly and preserve the last committed optimizer state.
These scale bounds are operational limits, not modeling assumptions.

Initialization, optimization, independent objective evaluation, and q sampling
use four separate PCG64 streams from a seeded `SeedSequence`. Opening a plot or
evaluating the objective cannot change subsequent optimization. A checkpoint
contains all three continuing random streams, mu, rho, Adam moments, iteration
counts, objective histories and status. Pause/resume equality is tested in the
same runtime; different numerical backends are not promised identical outcomes.
PCG64's 128-bit state and increment are exported as decimal strings, so a
JavaScript `JSON.parse`/`JSON.stringify` round trip cannot round them to 53-bit
integers. Tests exercise that actual JavaScript round trip and compare all
three streams after restoration; exact legacy native integer states are also
accepted on import.

`optimization_elbo` records the noisy pre-update batch objective. `evaluate`
uses a separate random stream and reports the independent estimate together
with `std(log_target_z)/sqrt(n)` Monte Carlo standard error; entropy is analytic.
History may decrease between iterations. A stable objective is not evidence
that q has captured uncertainty or all modes. The label ELBO describes a lower
bound on log evidence only when the target is a correctly normalized log joint;
an arbitrary unnormalized density does not yield model evidence.

For `N(m, Sigma)`, the exact reverse-KL optimum has mu=m and diagonal variances
`1/diag(inv(Sigma))`, not the marginal variances `diag(Sigma)`. With unit marginal
variances and correlation 0.95, both optimal VI variances are 0.0975. Tests fit
three predefined seeds, check means and variances, compute exact Gaussian KL,
and compare the independent objective against negative KL. A transformed
Beta(9,5) scalar objective is separately checked by adaptive quadrature.
Run multiple requested initializations independently and report each actual
outcome. Variational draws have no MCMC R-hat or ESS; they are draws from q,
whose error relative to the posterior is a separate question.

## Recorded independent inference comparison

`scripts/reference_inference.py` compares Posterior's compiled Gaussian and
Beta–Binomial models against the development-only **emcee 3.1.6** ensemble
sampler, with independently coded analytic densities. Its compact report is
`reports/reference-inference.json`; every retained draw is preserved in
`reports/reference-inference-draws.npz`, with array shapes and SHA-256 recorded
in the report. No inference package executes production Posterior algorithms.

Each Posterior run uses four chains, 300 warmup transitions and 1,000 retained
draws per chain. The measured Gaussian NUTS pilot used four chains with 50
warmup and 50 retained draws: 0.580 seconds and 5,285 gradient evaluations.
The full Gaussian NUTS comparison took 7.624 seconds for 67,654 evaluations;
the Beta NUTS comparison took 6.802 seconds for 27,092 evaluations. These are
native Python 3.14.3 / NumPy 2.4.2 / SciPy 1.17.1 measurements on the recorded
arm64 macOS host, not browser measurements or promises for other hardware.
emcee evaluates a hand-coded density while Posterior evaluates its graph and
gradients, so these timings are not a pure comparison of sampler overhead.

The audit found a real failure worth keeping. For the highly correlated
Gaussian, fixed **eight-step HMC** has individual-coordinate R-hat at most
1.0046, but the analytically known narrow direction `(x-y)/sqrt(2)` has R-hat
**1.1476** and bulk ESS **20.25**. The report therefore ends with
`completed_with_failed_checks`. Principal-direction diagnostics were added
during the audit after inspecting this discrepancy; thresholds were not
relaxed. HMC's fixed integration duration can resonate with Gaussian motion.
This run is retained in full, not replaced by a favorable random seed.

A documented follow-up keeps the same seed, warmup and draw budget and changes
only the number of HMC integration steps to **seven**. It takes 5.126 seconds
and 46,816 evaluations; the narrow-direction R-hat is **1.0011** and bulk ESS
**1,339.62**. Its posterior mean is `[-0.01060, -0.00803]` against `[0,0]`.
This supports sensitivity to trajectory length; it does not guarantee that
seven steps works on other targets. Gaussian NUTS, Beta HMC/NUTS, and the
independent reference runs pass the report's declared checks. emcee walkers
are coupled, so their diagnostics use its ensemble autocorrelation estimate,
not independent-chain R-hat.

Reproduce with `.venv/bin/python scripts/reference_inference.py`. It returns
exit code 1 when the preserved failure checks occur, after writing all results.
This experiment is not part of the site's normal build or the fast unit tests.
