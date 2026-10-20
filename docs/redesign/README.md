# Posterior: a probability atlas

The redesign replaces the former three-column dashboard with a figure-led workbench. The warm paper, indigo, sage, ochre, and rose palette remains; Georgia and the existing system sans-serif stack remain. No external font, UI framework, animation library, service, or engine dependency was added.

## Direction and product audit

Three structures were considered: a step-by-step inference wizard, a full-screen instrument, and an atlas of probability experiments. The atlas keeps assumptions, a real distribution, and its computational evidence accessible together while allowing five distinct questions to share a coherent interface. A wizard would unnecessarily separate the figure from its assumptions; a full-screen instrument would obscure interpretation and supporting evidence.

The old interface placed Run inference below the first viewport (about 1,175 pixels down at 1440 × 1000). On mobile, settings preceded the main figure, whose top was about 1,249 pixels down. Small type and three competing columns obscured the distinction between independent references, live samples, and diagnostics.

The portfolio was inspected in a browser, including its opening and work section, and through its relevant public source. It provided a standard for hierarchy, responsive composition, focused interactions, and finish; none of its assets, motifs, composition, or animation choreography was copied. [Reference audit](reference-audit.md).

## Structural changes

- A question-led model strip replaces the stacked preset list. The five experiments remain individually discoverable.
- The distribution becomes the main figure, beside an experiment panel with a prominent Run action. Its caption distinguishes references, retained draws, and recorded integration positions.
- Settings are divided into model assumptions and inference method. The JSON editor and integration budgets remain available through disclosures.
- Run progress and comparisons sit beneath the figure. Traces, rank plots, variational objectives, summaries, and numerical diagnostics have a dedicated evidence section.
- Local saving, imports, and portable exports form a separate archive section. Native dialogs provide keyboard focus containment and Escape dismissal.
- Tablet settings reflow below the full-width figure. Mobile uses a horizontally navigable model strip, an early Run action, direct settings/diagnostic links, and another synchronized Run action after editing settings.

## Visualization and interaction

Canvas axes use readable type and responsive tick density. The Beta reference includes an exact mean marker and explicitly labeled central 90% reference shading. Regression retains separate mean-response and predictive bands. Two-dimensional contours remain equal-density levels; funnel reference points remain generative samples. Trajectories retain branch-aware NUTS geometry and distinguish integration positions from posterior draws.

Pointer, touch, and keyboard inspection report actual nearest plotted values. Separate chains and independent variational fits are never pooled for display. Plot marks are clipped to their scientific domains. Keyboard focus is visible; reduced motion removes transitions and makes replay jump to the final recorded position. Progress and replay follow real computation rather than decorative animation.

## Critical review and correction

The first complete pass was inspected before further refinement. [Desktop first pass](review/first-pass-desktop.png) and [mobile first pass](review/first-pass-mobile.png) preserve that stage.

The final initial action starts at about 420 pixels on desktop and 492 pixels on mobile. The mobile figure starts at about 624 pixels, compared with 1,249 pixels before. No horizontal page overflow was found at any of the four checked widths.

The correction pass tightened the introduction and figure header so the distribution appears earlier, shortened the initial mobile controls, added direct settings/diagnostic links and a synchronized mobile Run action, and removed legends for chains before any run exists. It also repaired selected-state accessibility, transitions between VI and MCMC diagnostic views, stale transition details after invalidation, and matching runtime-failure labels on both Run actions, and accessible names for every dialog.

## Before and after

All matching captures use the same viewport dimensions. The result comparison uses a real Beta NUTS run: seed 42, four chains, 200 warmup and 200 retained draws per chain. A completed run is not described as converged; the recorded default run reports chain disagreement.

| State | Before | After |
| --- | --- | --- |
| Initial, 1440 × 1000 | [Before](before/desktop-initial.png) | [After](after/desktop-initial.png) |
| Initial, 1280 × 900 | [Before](before/desktop-1280-initial.png) | [After](after/desktop-1280-initial.png) |
| Initial, 768 × 1024 | [Before](before/tablet-initial.png) | [After](after/tablet-initial.png) |
| Initial, 390 × 844 | [Before](before/mobile-initial.png) | [After](after/mobile-initial.png) |
| Real Beta result, 1440 × 1000 | [Before](before/desktop-result.png) | [After](after/desktop-result.png) |
| Recorded transition, 1440 × 1000 | [Before](before/desktop-secondary.png) | [After](after/desktop-secondary.png) |

Additional captures cover [regression](after/desktop-regression.png), [variational inference](after/desktop-variational.png), [advanced controls](after/desktop-advanced.png), [the guide](after/desktop-guide.png), and [a mobile result](after/mobile-result.png). Captures contain real computations or explicitly labeled empty/reference states.

## Verification

- Native regression suite: **178 passed**, 60.12 seconds in the local environment. Python engine and worker sources are unchanged.
- Existing browser acceptance: **18 passed**. It covers all five models, native/Pyodide parity, pause/resume, transactional recovery, cancellation, stale-response rejection, CSV and JSON imports, exports, replay, comparisons, unavailable diagnostics, runtime failure/retry, and static asset errors. [Production browser report](../../reports/browser/report.json).
- The redesign suite: **10 passed**. It checks the four responsive widths, primary-action visibility, unique IDs and label targets, keyboard plot inspection, native-dialog focus return, actual default inference, reference recomputation, advanced controls, regression/VI results, and reduced motion. [Redesign report](after/report.json).
- Both browser suites run in CI after the production build. `npm run test:redesign` starts a preview when no URL is supplied; `POSTERIOR_URL` can point it at the deployed `/Posterior/` path.

The Python numerical semantics, source archive, pinned runtime, datasets, experiment schema, RNG behavior, worker acknowledgment contract, and scientific limits remain intact. Previous statistical failures remain preserved in the evaluation reports.

## Limits

Browser verification uses Chromium on macOS and in Linux CI. Safari, Firefox, physical touch devices, and background-tab behavior remain unverified. Touch handlers and responsive geometry are implemented; viewport emulation is not a physical-device test. Short regression and VI screenshot runs establish execution, not posterior accuracy. No new statistical-quality claim follows from the redesign.
