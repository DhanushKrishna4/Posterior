# Version 1 model language

Every preset, builder model, and editor document uses this representation. See `examples/` for complete JSON models. Model text is parsed as JSON only; it never goes through Python `eval` or `exec`.

```json
{
  "version": 1,
  "name": "A normal location",
  "parameters": [{"name": "location", "shape": [], "constraint": {"type": "real"}}],
  "data": {"observed": [0.2, 1.0, 0.7]},
  "nodes": [],
  "factors": [
    {"name": "prior", "distribution": "normal", "value": "location", "params": {"loc": 0, "scale": 2}},
    {"name": "likelihood", "distribution": "normal", "value": "observed", "params": {"loc": "location", "scale": 1}}
  ],
  "derived": []
}
```

Names begin with a letter and contain at most 48 letters, digits, or underscores. Names are unique across parameters, data, and deterministic nodes. Factor names are unique within the factor list. A string argument is a reference; a number or numeric array is a constant. Scalar parameters have `shape: []`, vectors have `shape: [n]`. Supported constraints are `real`, `positive`, and `interval` with finite increasing `lower`/`upper` bounds.

Nodes have `name`, `op`, and `args`, with these operation-specific fields:

| Operation | Arguments and fields |
| --- | --- |
| `add`, `sub`, `mul`, `div` | Two broadcast-compatible arguments |
| `pow` | One argument and literal `power` in [−16,16] |
| `neg`, `exp`, `log`, `log1p`, `sigmoid`, `softplus` | One argument |
| `sum`, `mean`, `logsumexp` | One argument; optional fixed `axis`, `keepdims` |
| `index` | One argument; fixed integer `index`, or list of coordinate integers |
| `reshape` | One argument; positive `shape`, preserving element count |
| `dot`, `matvec` | Equal-length vectors, or matrix and matching vector |
| `stack` | Fixed list of equally shaped arguments; optional `axis` |
| `logpdf` | `distribution`, `value`, `params`; produces elementwise log density |

Factor density arrays are summed. A factor may alternatively specify `log_density: "node_name"`; this enables stable fixed mixtures through weighted log densities, `stack`, and `logsumexp`. Such a factor is not automatically a normalized distribution. `density_kind` documents the intended interpretation; the compiler cannot prove normalization of arbitrary compositions.

Distribution parameters are explicit and validated: `normal(loc,scale)`, `half_normal(scale)`, `beta(alpha,beta)`, `bernoulli_logits(logits)`, `binomial(trials,prob)` or `binomial(trials,logits)`, and `multivariate_normal(loc,cov)`. Beta shapes, binomial trial counts, and covariance must be constant. Bernoulli/binomial values must be observed constants, never latent variables. Covariance must be finite, symmetric, and positive definite. Unknown distribution parameters are rejected.

`derived` lists named quantities for reporting. An optional `report` list chooses reporting nodes, allowing noncentered funnel base variables to be displayed on common `(y, children)` axes. Reporting names must be distinct, with no more than 128 scalar quantities. Reported quantities do not add density terms or Jacobians.

The compiler topologically orders a static graph, validates names/cycles, shapes and broadcasting, support and constant hyperparameters, finite inputs, and resource bounds. Array results are bounded before broadcast allocation. A numerically invalid proposal returns a nonfinite-target signal and is handled by the sampler; parameters and sampler gradients are never silently clipped. Invalid model structure produces an actionable validation error.

Only bundled regression and Beta–Binomial prediction specifications are supported. Custom graphs can report arbitrary supported deterministic quantities, but do not inherit an oracle or predictive generator by copying a preset label. Data has no hidden preprocessing: regression CSV imports record the selected values directly.
