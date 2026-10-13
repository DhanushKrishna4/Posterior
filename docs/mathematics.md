# Density, coordinates, and approximation

Let `theta=T(z)` map unconstrained coordinates into model parameters. The engine evaluates

```
log target_z(z) = sum(named log-joint contributions at T(z)) + log |det J_T(z)|.
```

Density constants are retained in built-in distributions. A normalized target integrates to one; a log joint includes observed data and generally integrates over parameters to the marginal likelihood; a posterior divides that joint by its normalizing constant. A generic composed factor may be unnormalized. A log-density evaluation is never called model evidence.

Identity has zero log-Jacobian. For positive parameters, `theta=exp(z)` and `log J=z`. For a finite interval `(a,b)`, `theta=a+(b-a)*sigmoid(z)` and `log J=log(b-a)-softplus(-z)-softplus(z)`. Stable log probabilities from sigmoid nodes avoid cancellation at extreme finite logits. Floating-point transformed values may round to a boundary at extreme coordinates; no clipping is applied. Nonfinite arithmetic is reported rather than repaired by altering a target.

The default Beta–Binomial transformed posterior is proportional to

```
9 log(sigmoid(z)) + 5 log(sigmoid(-z)).
```

The extra powers compared with the model-space beta density come from its one transform Jacobian. Finite differences and numerical integration validate this change of variables.

The noncentered funnel instead declares independent base variables: `y~Normal(0,3)`, `z_i~Normal(0,1)`, and derives `x_i=exp(y/2)*z_i`. Its integration variables are already `(y,z)`. The deterministic expression defining `x` adds no Jacobian. The centered model declares `(y,x)` and uses conditional normal density terms.

For diagonal preconditioning, `z=Sx` is a separate change of working coordinates. The sampler evaluates `log target_x=log target_z(Sx)+sum(log S)` and `gradient_x=S*gradient_z`, with unit kinetic energy `0.5*r·r`. Every trajectory and U-turn check is in the recorded working coordinates. Scales remain fixed during a transition.

VI uses `z=mu+exp(rho)*epsilon`. Its objective is the expected transformed log joint plus the analytic Gaussian entropy. Reparameterized gradients are `E[grad_z]` for `mu` and `E[grad_z*exp(rho)*epsilon]+1` for `rho`. Adam updates use fresh optimization noise. Objective evaluation uses a separate stream and reports its standard error; prediction and variational sampling use separate streams too.

For a Gaussian target with mean `m` and covariance `Sigma`, reverse-KL mean-field optimum is `mu=m`, `variance_i=1/(Sigma^-1)_ii`. The Gaussian KL is computed independently as

```
0.5 * [tr(Sigma^-1 D) + (mu-m)' Sigma^-1 (mu-m) - d
       + log det Sigma - log det D].
```

At correlation 0.95 and unit marginal variances, `D_ii=0.0975`. This is approximation bias even at perfect optimization, not Monte Carlo uncertainty.

References: [Stan constraint transforms](https://mc-stan.org/docs/reference-manual/transforms.html), [ADVI](https://jmlr.org/papers/v18/16-107.html), and [Neal funnel discussion](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html). Sampler derivation and exact NUTS variant are in [samplers.md](samplers.md).
