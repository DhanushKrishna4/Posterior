# Redesign benchmark and preservation audit

Source review performed on 19 October 2026. The reference is a standard of execution only. Posterior must retain its own palette, scientific purpose, composition, imagery, and interaction vocabulary.

## Reference access and inspected source

The public [portfolio repository](https://github.com/DhanushKrishna4/DhanushKrishna4.github.io) was accessible through GitHub and a temporary read-only clone. The inspected revision was `d9b70c3364c5c7399bedf3e86c46f43d027a8887`. No assets or application code were copied into Posterior.

Relevant source reviewed:

- `src/App.tsx`: page composition, section relationships, font-ready geometry refresh, effect cleanup.
- `src/components/Nav.tsx` and `Menu.tsx`: deliberate navigation priority, open/closing states, keyboard access, focus return, background isolation.
- `src/components/Hero.tsx`, `Core.tsx`, `Work.tsx`, `Record.tsx`, and `Contact.tsx`: hierarchy across the full experience, secondary interactions, graphics fallback, responsive content relationships.
- `src/styles/index.css`: shared spacing/type/motion tokens, fluid proportions, focused/hover states, explicit responsive arrangements, reduced-motion rules.
- `src/lib/motion.ts`, `scroll.ts`, and `inert.ts`: coordinated motion, native reduced-motion scrolling, preventing reveal states from stranding content, restoration of focus and interaction.
- `src/lib/sculpture-layout.ts`: layout-specific visual framing for phone, portrait tablet, landscape tablet, and wider desktop contexts.

The browser inspection of the live reference is a separate part of the redesign review. This document records the source audit rather than claiming browser checks from source alone.

## Transferable standards

1. **Hierarchy must be authored.** Major content has a clear scale and proportion relationship; the primary action is evident. For Posterior, the live distribution and inference action should carry that hierarchy.
2. **A visual system includes behavior.** Shared spacing, typography, line treatment, focus, status, and transition choices should extend through advanced settings and saved experiments.
3. **Responsive means recomposed.** Different viewport shapes need different content relationships and useful graphic dimensions. A smaller desktop layout is insufficient.
4. **Motion must preserve access.** State transitions should support orientation. Reduced motion must retain content and complete states without requiring travel or delayed controls.
5. **Keyboard behavior is part of finish.** Visible focus, sensible focus placement, reliable Escape behavior, and returning focus after closing a dialog matter as much as hover styling.
6. **Graphics must survive real conditions.** Font loading, resizing, touch input, unavailable capabilities, and initialization failure require considered states.
7. **Detail must continue beyond the opening screen.** The workspace, secondary views, exports, empty/error states, and supporting explanation deserve the same attention as the initial composition.

These are implementation standards. The reference's section order, mark, layered surfaces, graphic motifs, assets, and animation choreography are not design inputs for Posterior.

## Posterior contracts to preserve

Reviewed `web/client.js`, `web/worker.js`, `web/main.js`, `web/plots.js`, and `scripts/browser-check.mjs`.

- Keep one persistent Python worker. Navigation, disclosures, and plot tabs must not create additional workers or reload the runtime.
- Preserve generation filtering and model-change invalidation. Responses from an earlier configuration must not overwrite the current state.
- Preserve the bounded request queue, incremental delta merge, committed checkpoints, and snapshot acknowledgment. The worker deliberately waits for an acknowledgment between committed transitions; changing that flow can break recovery or overload rendering.
- Pause/cancel operate between completed transitions. Recovery recreates the worker from the last completed checkpoint and discards unfinished work. The interface must not imply immediate interruption inside a transition.
- Preserve model identity and references derived by Python when importing a checkpoint. User-provided envelope labels or reference quantities are not authoritative.
- Retain all five presets, NUTS/HMC/VI, custom graph validation, regression CSV import, all integration/resource settings, initialization choices, recorded transition selection/replay, comparisons, local save/load/delete, and JSON/CSV/PNG exports.
- Keep independent VI fits separate; MCMC diagnostics are not applicable to VI. An objective plateau is not evidence of accurate uncertainty.
- Continue distinguishing analytic/generative references, retained draws, warmup, intermediate integration positions, and unavailable diagnostics. Contours are equal-density levels, not claimed probability-mass regions.
- Keep comparison compatibility checks, chain-specific mode occupancy and narrow-funnel coverage, numerical divergences, depth-limit events, ESS, R-hat, MCSE, and uncertainty summaries.
- Preserve static `/Posterior/` asset resolution, pinned local runtime dependencies, reproducible seeds, and the inspection API used by browser verification.

## Markup replacement and verification pitfalls

The current application binds many event handlers directly to element IDs and `data-*` attributes. Replacing the markup safely requires retaining those hooks or systematically updating every caller; a missing element can abort boot before the runtime starts. Labels such as `label[for="draws"]`, `.field` ancestors, `.lower-plot`, and `.progress-track` are also behavioral selectors, not merely styling hooks.

The browser suite currently contains 18 checks spanning real Pyodide execution, five-model native parity, NUTS and VI workflows, checkpoint recovery, cancellation/stale responses, imports, exports, comparisons, replay, small-screen behavior, runtime failure/retry, and browser errors. Its settings interactions assume particular disclosures and visible controls. When navigation changes, update the test to exercise the new route to those controls; do not weaken the scientific assertions or force hidden controls to pretend the interface works.

Canvas plots derive their pixel dimensions from the visible layout and cap device pixel ratio at 2. A canvas rendered while its panel is hidden has no useful size. Re-render after activating a panel or changing its dimensions; retain enough plot height at narrow widths. Geometry also needs refreshing after fonts settle. Keep raster export an actual image of the current plot.

The asset-origin check expects all browser assets to come from the site origin. Any typography change should respect the self-hosted runtime/site contract. The existing redraw scheduler limits plotting to roughly one update every 120 ms; visual motion should not add expensive work to each sampler transition.

The existing mobile check verifies 390-pixel overflow and native-dialog Enter/Escape behavior; it is not sufficient to substantiate the redesign's four-width visual requirement. Inspect 1440, 1280, 768, and 390 pixels, equivalent before/after states, loading/error states, real results, secondary views, and keyboard navigation separately.
