# Browser implementation and verification

The production site is built for the case-sensitive `/Posterior/` subpath. Run:

```sh
npm ci
npm run vendor
npm run build
npm run test:browser
npm run preview
```

The browser check starts its own localhost preview server on port 4178 unless `POSTERIOR_URL` is set. It uses Playwright's installed Chromium. On a development Mac it can also discover an existing Chromium cache. Set `PLAYWRIGHT_EXECUTABLE` to an explicit executable when needed. CI installs Chromium with `npx playwright install --with-deps chromium`. Native parity checks use `PYTHON`, `.venv/bin/python` when available, or `python3`.

`reports/browser/report.json` is the machine-readable record of the most recent executed suite. Each gate reports passed or failed; execution-only short runs are not statistical quality claims. The suite records the exact runtime environment, model, settings, completed work, wall time, gradient work, and retained draws. Screenshot artifacts show ready, VI, trajectory, comparison, and narrow-screen states.

## Runtime assets

`scripts/runtime-lock.json` pins SHA-256 hashes and byte counts for Pyodide 0.27.7, its Python 3.12 standard library, NumPy 2.0.2, SciPy 1.14.1, and OpenBLAS 0.3.26. The required runtime totals about 34.8 MiB uncompressed transfer bytes before HTTP compression. `npm run vendor` downloads only these files and checks all hashes, including core WASM and JavaScript. The build rechecks the hashes. Runtime downloads happen during preparation; the shipped application serves every runtime asset itself and performs inference locally.

The Python engine is a deterministic ZIP: sorted Python source files, a fixed ZIP member timestamp, and fixed member permissions. No compiled native package is silently substituted for the engine. Vite emits the module worker and resolves its URL beneath the configured site base.

## Worker contract

One persistent worker owns one Python `Experiment`. Every synchronous Python step completes or fails transactionally before the event loop services another message. The worker yields using `setTimeout` between transitions. It sends an incremental committed-state update after each completed step, and waits for the main thread to merge and acknowledge it before starting another transition. This bounds outgoing updates to one and preserves a complete checkpoint outside the worker. Full snapshots reconcile results and refresh diagnostics every 25 transitions and on the terminal result; the snapshot includes the retained count associated with cached diagnostics. Replay does not consume inference randomness.

Pause and cancel are cooperative between completed transitions. Recover worker terminates the worker immediately, starts a new Pyodide instance, and restores the last checkpoint held on the main thread. The incomplete transition is discarded. The restored run is paused until explicitly resumed. Generation IDs reject stale messages; the main-thread request queue is capped at 16. A stable owned PyProxy for the JSON bridge is the only retained Python proxy.

RNG state remains lossless across Python JSON, JavaScript, storage, and Python restore: the two PCG64 uint128 words are encoded as decimal strings. Model hashing normalizes integral floating-point literals so JSON conversions such as `2.0` to `2` preserve model identity. The browser suite checks exact retained draws and RNG state against uninterrupted execution in the same runtime. Node JSON-round-trip tests also verify native checkpoints through the JavaScript serialization boundary.

## Plot semantics

- Beta–Binomial shows separately computed prior, scaled likelihood shape, analytic posterior, and normalized retained-draw histograms.
- Regression shows analytic central 90% mean-response and predictive bands separately. Sampled regression lines use retained coefficients.
- Gaussian and mixture contours are equal-density contours of the known two-dimensional target. They are not labeled probability-mass regions.
- The six-dimensional funnel view is a sample-based marginal projection onto original model-space y and the first child. Independent generative reference draws are visually separate.
- Custom graphs receive sample-based marginal plots with no fabricated oracle.
- Trajectory paths are recorded working coordinates. NUTS links restart from the actual backward or forward edge at each expansion; depth-first traversal jumps are never connected as fictitious leapfrog motion. Excluded leaves are hollow. The retained endpoint uses `selected_position`, not the last integration leaf.
- Replay reads recorded positions and metadata; it does not consume Python randomness or change inference.
- A bounded history keeps up to eight prior runs. A previous alternative method overlays only when its model hash exactly matches the current run. Its label and summary stay separate.
- VI fits are alternatives, not pooled chains. Their fit selector, independently estimated ELBOs, and objective histories remain distinct. MCMC R-hat and ESS are marked not applicable.

Local saves are capped at eight experiments. CSV imports are capped at 250 kB, 20 columns, and 1,000 observations, with explicit predictor/response selection and finite validation. JSON imports are capped at 128 MB and validated again by Python before restore. Exports store complete results once inside the checkpoint rather than duplicating draw arrays. PNG export includes the actual rendered plot. The interface is responsive, keyboard operable, and respects reduced-motion preferences.
