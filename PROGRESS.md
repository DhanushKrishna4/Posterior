# Posterior implementation and verification

Local implementation delivered 2026-10-12. The initial workspace was empty.
Public release was authorized on 2026-10-13 and verified on 2026-10-14.
The public repository is [DhanushKrishna4/Posterior](https://github.com/DhanushKrishna4/Posterior),
with the [live workbench](https://dhanushkrishna4.github.io/Posterior/) hosted on GitHub Pages.

## Implemented milestones

1. Installable Python package, CLI, versioned JSON schema and validated static graph.
2. Float64 reverse-mode autodiff, probability primitives and parameter transforms.
3. Analytic/finite-difference gradients, stable primitives and change-of-variable checks.
4. Leapfrog, HMC acceptance, momentum handling and repeated rejected draws.
5. Original Hoffman–Gelman Algorithm 6 slice NUTS with controlled small-tree tests.
6. Dual averaging, explicit working coordinates, variance windows and frozen adaptation.
7. Mean-field VI, Adam, independent random streams and objective evaluation.
8. Rank/folded split R-hat, bulk/tail ESS, raw-mean MCSE and ArviZ reference fixtures.
9. Five model graphs, independent references, predictive simulation and failure demonstrations.
10. Persistent Pyodide worker, atomic transitions, incremental checkpoints and exact same-runtime resume.
11. Probability plots, actual trajectory replay, comparisons, separate VI fits, CSV/JSON import, local saves and exports.
12. Numerical evidence, raw results, HTTP browser checks, documentation and static Pages configuration.

## Verified gates

| Gate | Result |
| --- | --- |
| Full native suite | **178 passed** in 51.95 s |
| HMC/NUTS statistical validation | **6/6 passed**, predefined seeds on Gaussian, Beta–Binomial and regression |
| Native end-to-end evaluation | **12/12 completed**, complete raw draws/checkpoints retained |
| Native/Pyodide primitives | 10 fixtures across five presets; maximum error `4.44e-16`, allowed `1e-9 + 1e-10*abs(reference)` |
| Diagnostic reference | ArviZ 0.21.0 live comparisons and fixtures at `2e-10`; invalid/constant cases explicitly unavailable |
| Python distribution | Source archive and universal wheel built; isolated wheel import from `/tmp` ran all 15 preset/algorithm combinations |
| Browser and release | **18/18 passed; artifact audit passed.** Authoritative counts and hashes in [browser report](reports/browser/report.json) and [build audit](reports/build-verification.json) |

The production browser suite covers all presets, sampling/VI, actual controls,
unavailable diagnostics, pause/resume, cancellation, model changes, stale
responses, worker recovery, native/browser checkpoint imports, local deletion,
CSV validation, PNG/CSV/JSON export, runtime retry, keyboard controls and a
390 × 844 layout. Long-transition cancellation uses 32 dimensions, 90 graph
nodes and 200 HMC steps; cancellation commits only whole transitions and forced
recovery discards unfinished work. The final run also checks custom-model
recovery, funnel parameterization and mixture initialization comparisons.

## Measurements and evidence

Apple M3, 8 GiB RAM, sequential CPU. Four-chain native Beta NUTS with 200 warmup
and 200 retained draws per chain took **1.99 s**. The browser default completed
1,600 transitions and 800 retained draws in about **16 s** including UI work.
Exact settings and runtime differences are recorded; these are not pure
language-overhead comparisons. Native peak process RSS was about **145 MiB**;
total browser memory was not measured.

See [the evaluation report](reports/README.md),
[native results](reports/native-evaluation.json),
[sampler validation](reports/sampler-validation.json), and
[independent reference comparison](reports/reference-inference.json).

## Preserved failures and remaining limits

- Eight-step HMC on the correlated Gaussian fails its narrow-direction quality gate: R-hat 1.1476 and bulk ESS 20.25. The full failed run remains available. A same-seed seven-step followup passes; this is sensitivity to path length, not a universal recommended setting.
- The centered funnel quick run misses narrow-region mass. Noncentering improves this particular run; success is not guaranteed for arbitrary short runs.
- Saved separated-mixture runs make no mode crossings. Equal pooling of trapped chains does not recover 70:30 weights. Independent VI fits stay separate.
- At Gaussian correlation 0.95, optimal mean-field variances are 0.0975 while true marginal variances are 1. Accurate optimization can still understate uncertainty.
- Safari, Firefox, physical touch, background-tab behavior and browser heap growth are unverified. Cross-runtime primitive parity is tested; identical full chains across builds are not promised.
- GitHub Actions passed all 178 Python tests and 18 browser checks, and deployed successfully. The hosted Pages app separately passed 18/18 browser checks. See [deployment evidence](reports/deployment.json) and [hosted checks](reports/hosted/report.json).

## Reproduction commands

From this directory using the existing environment:

```sh
.venv/bin/python -m pytest -q
npm run build
npm run test:browser
.venv/bin/python scripts/verify_build.py
npm run preview
```

Open `http://127.0.0.1:4173/Posterior/`. Fresh-machine setup, pinned dependencies
and runtime vendoring are documented in [README](README.md).

To regenerate numerical evidence, run sequentially:

```sh
.venv/bin/python scripts/evaluate.py --pilot
.venv/bin/python scripts/validate_samplers.py
.venv/bin/python scripts/reference_inference.py
.venv/bin/python scripts/evaluate.py
```

The independent-reference command returns exit status 1 when the preserved
HMC gate fails, after saving all results. Run these commands separately;
an `&&` chain would skip the following evaluation. Research benchmarks never
execute during a normal site build.
