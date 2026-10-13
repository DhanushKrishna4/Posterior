# Recorded evaluation

These artifacts contain actual CPU computations, including unsuccessful exploration. A completed run means its requested work finished; it does not mean its posterior estimate is trustworthy. Numerical validation, short illustrative runs, independent reference comparisons, and browser execution checks answer different questions.

## Evidence and environment

| Artifact | What it records |
| --- | --- |
| [hardware.json](hardware.json) | Apple M3, 8 GiB RAM, arm64; local CPU |
| [native-pilot.json](native-pilot.json) | Measured small Beta–Binomial pilot before the native evaluation |
| [native-evaluation.json](native-evaluation.json) | Twelve actual preset outcomes, settings, costs, diagnostics, model/source hashes and raw-artifact hashes |
| [raw/](raw/) | Complete compressed JSON snapshots and checkpoints for all twelve native cases |
| [sampler-validation.json](sampler-validation.json) | Six fixed-kernel distributional checks, exact oracles, full raw draws, and a preserved HMC failure |
| [reference-inference.json](reference-inference.json) | Independent emcee comparison and the retained Gaussian HMC failure investigation |
| [reference-inference-draws.npz](reference-inference-draws.npz) | Every retained reference-comparison draw; array shapes and SHA-256 are in the JSON report |
| [browser/report.json](browser/report.json) | HTTP browser checks, separate Pyodide measurements, tolerances and failures |
| [build-verification.json](build-verification.json) | Current-source identity, raw checkpoint hashes, runtime asset integrity and reproducible production archive |
| [browser/default-pilot.json](browser/default-pilot.json) | Complete default browser pilot results |

Native measurements use Python 3.14.3, NumPy 2.4.2 and SciPy 1.17.1. Browser measurements use Pyodide 0.27.7, Python 3.12.7, NumPy 2.0.2 and SciPy 1.14.1. All inference uses float64. Recorded elapsed times describe this host and run; they are not runtime guarantees. Machine-readable reports are authoritative when regenerated.

## Short native preset runs

The twelve-case evaluation uses seed 42, four sequential/round-robin chains, 200 warmup transitions and 200 retained draws per chain. NUTS depth is capped at 7, target acceptance is 0.8, and each run has limits of 120 seconds and 500,000 gradient evaluations. HMC uses ten integration steps. Four independently initialized VI fits each use 1,000 Adam updates, batch size eight, learning rate 0.03, and 200 final approximation draws. VI objective evaluations use a separate random stream; no MCMC diagnostics are applied to these draws.

The preceding native Beta pilot used four chains with 25 warmup and 25 retained draws each: **0.283 seconds** including the run loop and final snapshot, **1,059 gradient evaluations**, and R-hat **1.0742**. Its small budget is intentionally insufficient for a reliable estimate.

| Preset / method | Run + snapshot seconds | Gradient evaluations | Retained divergences / depth-limit events | Recorded assessment |
| --- | ---: | ---: | ---: | --- |
| Beta–Binomial / NUTS | 2.056 | 7,775 | 0 / 0 | Chains disagree |
| Beta–Binomial / VI, four fits | 9.526 | 42,240 | Not applicable | Approximation only |
| Regression / NUTS | 1.915 | 9,882 | 0 / 0 | No disagreement detected |
| Regression / VI, four fits | 6.956 | 42,240 | Not applicable | Approximation only |
| Correlated Gaussian / HMC | 2.110 | 19,217 | 0 / 0 | Chains disagree |
| Correlated Gaussian / NUTS | 2.236 | 19,196 | 0 / 0 | Chains disagree |
| Correlated Gaussian / VI, four fits | 4.073 | 42,240 | Not applicable | Approximation error quantified below |
| Centered funnel / NUTS | 11.664 | 61,627 | 1 / 60 | Chains disagree |
| Noncentered funnel / NUTS | 2.721 | 13,591 | 0 / 0 | No disagreement detected |
| Mixture, all chains near left mode / NUTS | 2.879 | 10,132 | 0 / 0 | Missed the right mode despite reassuring diagnostics |
| Mixture, dispersed chains / NUTS | 2.830 | 10,001 | 0 / 0 | Chains disagree; incorrect pooled weights |
| Mixture / VI, four fits | 9.882 | 42,240 | Not applicable | Different fits remain near different modes |

The time column excludes initial model construction and includes final summarization; `stats.elapsed_seconds` separately measures active engine work. Warmup divergences are recorded separately and excluded from the table. Process peak RSS is cumulative across sequential cases, not an isolated per-model allocation measurement.

For the default ten-trial Beta–Binomial model, the exact posterior is Beta(9,5), with mean **0.642857**. Short-run NUTS estimates **0.647339**, with mean MCSE **0.008087**, R-hat **1.0126**, and bulk ESS **233.5**. Numerical proximity to the oracle does not erase the diagnostic warning. Regression uses 24 observations: NUTS coefficient means are **[0.692029, 1.297137]**, with absolute errors **[0.001377, 0.001042]** against the conjugate Gaussian oracle.

## Variational approximation error

The correlated Gaussian has zero mean, unit marginal variances and correlation 0.95. Its exact reverse-KL mean-field optimum has coordinate variances **0.0975**, while the target marginal variances remain **1**. The optimal reverse KL is **1.163951** nats, so an accurately optimized factorization still understates uncertainty substantially.

| Native VI fit | Fitted mean | Fitted coordinate variances | Exact reverse KL of fitted approximation |
| --- | --- | --- | ---: |
| 0 | [0.014604, −0.011650] | [0.100619, 0.085699] | 1.171637 |
| 1 | [0.017224, −0.016713] | [0.089569, 0.092071] | 1.172265 |
| 2 | [0.027092, −0.009070] | [0.105980, 0.097291] | 1.172321 |
| 3 | [0.034869, −0.043083] | [0.078195, 0.091777] | 1.206564 |

Independent final ELBO estimates and their Monte Carlo standard errors are stored for each fit; they are noisy measurements, not monotone progress scores. The separate [VI tests](../tests/test_variational.py) use three predefined seeds, an analytic Gaussian KL oracle and transformed scalar Beta quadrature. They also check independent RNG streams and exact checkpoint restoration through an actual JavaScript JSON round trip.

## Observed difficult-target failures

The six-dimensional funnel has the known marginal `y ~ Normal(0,3)` and is a no-data generative benchmark. In the centered run, estimated y mean is **1.5982**, SD **2.1259**, R-hat **1.0416**, and bulk ESS **38.46**. Fractions below y=−3 by chain are **[0.05, 0, 0, 0]**, versus the known probability **0.158655**. It records one retained divergence and 60 depth-limit events; these are different conditions.

The noncentered run gives y mean **−0.0368**, SD **2.9706**, R-hat **1.0002**, and bulk ESS **1,175.8**. Its narrow-region fractions are **[0.18, 0.14, 0.145, 0.14]**, with no retained divergence or depth-limit event. These are outcomes of this seed and budget, not a guarantee that every noncentered run succeeds. Both formulations are reported on the same original model axes.

The two-component mixture has weights **0.7 / 0.3** and exact x mean **−1.6**. With every chain initialized near the left mode, all four remain there, with **zero region transitions**. Their pooled x mean is **−4.0151** even though x R-hat is **0.9985**. Agreement among chains cannot reveal a mode they all missed.

With dispersed initialization, two chains remain left and two remain right, again with **zero region transitions**. Equal pooling produces x mean **0.00115** and left occupancy **0.5**, not the target weight **0.7**; x R-hat is **1.7314** and bulk ESS **6.44**. The four VI fits likewise settle near separate modes. Neither pooling these chains nor pooling the fitted approximations demonstrates recovery of the mixture.

## Fixed-kernel validation and independent reference

[Sampler validation](sampler-validation.json) passes all six required combinations of HMC/NUTS with Beta–Binomial, Gaussian and regression targets. Each uses seeds **[17,29,53,101]**, 100 burn-in transitions and 750 retained draws per chain, with fixed documented step sizes/scales. Declared gates require R-hat below 1.01, bulk ESS above 400, mean error below five raw-scale mean MCSEs, bounded covariance/quantile error, and no divergences. This tests fixed kernels; warmup adaptation has separate deterministic tests.

The same artifact retains an unsuccessful Beta HMC configuration with ten steps at step size 0.35. It fails R-hat, ESS and quantile gates, and its original absolute mean-error check. Its integration duration is consistent with a resonance near the target's local period. Passing checks for another configuration do not delete that failure.

[Independent reference comparison](reference-inference.json) uses development-only **emcee 3.1.6** with independently coded normalized densities. Posterior runs four chains with 300 warmup and 1,000 retained draws per chain, seeds **307** for Gaussian and **811** for Beta. emcee uses 16 walkers, 500 warmup steps and 2,000 retained steps, with separately recorded seeds. Its walkers are coupled, so the report uses ensemble autocorrelation estimates rather than independent-chain R-hat. Reference density-only timings are not directly comparable to compiled graph-and-gradient timings.

The reference report deliberately concludes **`completed_with_failed_checks`**. Gaussian HMC with eight fixed integration steps has per-coordinate R-hat at most **1.0046**, but its analytically known narrow direction `(x−y)/sqrt(2)` has R-hat **1.1476**, bulk ESS **20.25**, and tail ESS **98.94**. Principal-direction checks were added during inspection; thresholds were not relaxed and the original draws remain saved.

A documented follow-up changes only the HMC integration count to seven at the same seed and budget. It passes the recorded coordinate and principal-direction gates: narrow-direction R-hat **1.0011**, bulk ESS **1,339.62**, and mean **[−0.010604, −0.008031]** against [0,0]. Gaussian NUTS, Beta HMC/NUTS and the emcee reference runs pass their recorded gates. This supports sensitivity to fixed trajectory length, not a universal recommendation for seven steps.

## Browser measurements and interpretation

[The browser report](browser/report.json) verifies the actual static application over HTTP under the case-sensitive `/Posterior/` subpath. It records cold loading, native/Pyodide primitive parity, short NUTS/VI execution for every preset, persistence, pause/resume, cancellation/recovery, stale responses, CSV validation, trajectory replay, exports, responsive layout and runtime-loading errors. The final run passes **18/18 checks**; read each check's status and the final summary; successful short execution is not a statistical accuracy test.

The recorded default browser Beta pilot retains **800 draws** after 800 warmup transitions, takes approximately **15.60 seconds including the UI**, and uses **7,707 gradient evaluations**. Its posterior mean **0.643576** is close to the exact mean, but R-hat **1.0227** explicitly reports chain disagreement. Native/Pyodide log-density and gradient parity uses absolute tolerance **1e−9** and relative tolerance **1e−10**; this does not promise identical full stochastic trajectories across numerical backends.

## Reproduction and raw data

From the repository root, with development dependencies installed:

```sh
.venv/bin/python scripts/evaluate.py --pilot
.venv/bin/python scripts/evaluate.py
.venv/bin/python scripts/validate_samplers.py
.venv/bin/python scripts/reference_inference.py
npm run build
npm run test:browser
.venv/bin/python scripts/verify_build.py
```

The reference script writes its full report and then exits with code 1 when the preserved failed gates occur. The site build does not run these numerical workloads. Full native snapshots can be read with `json.load(gzip.open(path, 'rt'))`; reference arrays load with `numpy.load(path, allow_pickle=False)`. No retained samples are thinned. Follow the relative artifact paths and SHA-256 values in the reports to identify the exact observations, configurations, model/source versions and results behind a claim.
