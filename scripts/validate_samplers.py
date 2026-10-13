#!/usr/bin/env python3
"""Reproduce fixed-seed sampler checks and a real fixed-length HMC failure.

Run from the repository: .venv/bin/python scripts/validate_samplers.py
All computation is sequential, local and CPU-only. This small research report
is separate from the static-site build. Failed criteria stay in the artifact;
the script returns nonzero when a required validation case fails. No retries.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.stats import beta, norm

from posterior_lab import __version__
from posterior_lab.adaptation import coordinate_target
from posterior_lab.diagnostics import diagnose
from posterior_lab.examples import preset, reference
from posterior_lab.models import compile_model
from posterior_lab.samplers import hmc_transition, nuts_transition


SEEDS = [17, 29, 53, 101]
BURN_IN = 100
DRAWS = 750
CRITERIA = {
    "rhat_less_than": 1.01,
    "bulk_ess_greater_than": 400,
    "absolute_mean_error_less_than_raw_mean_mcse_multiple": 5,
    "maximum_standardized_covariance_absolute_error": 0.2,
    "maximum_quantile_absolute_error_in_marginal_sd": 0.25,
    "allowed_divergences": 0,
    "quantile_probabilities": [0.05, 0.5, 0.95],
}


def environment():
    source_names = ["samplers.py", "adaptation.py", "models.py", "autodiff.py", "diagnostics.py"]
    sources = {}
    for name in source_names:
        path = ROOT / "posterior_lab" / name
        if path.exists():
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "posterior_lab": __version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "rng": "NumPy PCG64", "arithmetic": "float64", "execution": "sequential CPU",
        "source_sha256": sources,
    }


def collect(name, algorithm, *, seeds=SEEDS, burn=BURN_IN, draws=DRAWS,
            step_size=0.2, steps=5, scale_regression=True):
    before = time.perf_counter()
    spec = preset(name)["spec"]
    model = compile_model(spec)
    truth = reference(spec)
    covariance = (np.diag(truth["variance"]) if name == "beta"
                  else np.asarray(truth["covariance"]))
    scales = (np.sqrt(np.diag(covariance)) if name == "regression" and scale_regression
              else np.ones(model.dimension))
    target = coordinate_target(model, scales)
    sampler = hmc_transition if algorithm == "hmc" else nuts_transition
    all_draws, all_z, chain_summaries = [], [], []
    for seed in seeds:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        q = np.zeros(model.dimension)
        samples, z_samples = [], []
        gradients = integration_steps = divergences = depth_limits = rejected_or_repeated = 0
        acceptance_total = 0.0
        depths, reasons = Counter(), Counter()
        phase_counts = {"burn_in_gradients": 0, "retained_gradients": 0,
                        "burn_in_divergences": 0, "retained_divergences": 0}
        for iteration in range(burn + draws):
            kwargs = {"steps": steps} if algorithm == "hmc" else {"max_depth": 7}
            q, meta = sampler(target, q, rng, step_size, **kwargs)
            gradients += meta["gradient_evaluations"]
            integration_steps += meta["integration_steps"]
            divergences += int(meta["divergent"])
            depth_limits += int(meta["max_depth_reached"])
            rejected_or_repeated += int(not meta["accepted"])
            acceptance_total += meta["acceptance"]
            depths[str(meta["depth"])] += 1
            reasons[meta["stop_reason"]] += 1
            phase = "burn_in" if iteration < burn else "retained"
            phase_counts[f"{phase}_gradients"] += meta["gradient_evaluations"]
            phase_counts[f"{phase}_divergences"] += int(meta["divergent"])
            if iteration >= burn:
                z = scales * q
                samples.append(model.report(z).tolist())
                z_samples.append(z.tolist())
        all_draws.append(samples)
        all_z.append(z_samples)
        chain_summaries.append({
            "seed": seed, "elapsed_seconds": time.perf_counter() - started,
            "gradient_evaluations": gradients, "integration_steps": integration_steps,
            "divergences": divergences, "max_depth_events": depth_limits,
            "rejected_or_repeated": rejected_or_repeated,
            "mean_adaptation_acceptance_statistic": acceptance_total / (burn + draws),
            "depth_histogram": dict(depths), "stopping_reasons": dict(reasons),
            "final_rng_state": rng.bit_generator.state,
            **phase_counts,
        })
    values = np.asarray(all_draws)
    flat = values.reshape(-1, model.dimension)
    diagnostics = diagnose(values)
    exact_mean = np.asarray(truth["mean"])
    marginal_sd = np.sqrt(np.diag(covariance))
    probabilities = CRITERIA["quantile_probabilities"]
    exact_quantiles = (beta.ppf(probabilities, 9, 5)[:, None] if name == "beta"
                       else exact_mean + norm.ppf(probabilities)[:, None] * marginal_sd)
    observed_mean = flat.mean(axis=0)
    observed_covariance = np.atleast_2d(np.cov(flat.T))
    observed_quantiles = np.quantile(flat, probabilities, axis=0)
    mcse = np.asarray([p["mcse_mean"] for p in diagnostics["parameters"]], dtype=float)
    mean_error = abs(observed_mean - exact_mean)
    covariance_error = abs(observed_covariance - covariance) / np.outer(marginal_sd, marginal_sd)
    quantile_error = abs(observed_quantiles - exact_quantiles) / marginal_sd
    gate = {
        "rhat": all(p["rhat"] is not None and p["rhat"] < CRITERIA["rhat_less_than"]
                    for p in diagnostics["parameters"]),
        "bulk_ess": all(p["ess_bulk"] is not None and p["ess_bulk"] > CRITERIA["bulk_ess_greater_than"]
                        for p in diagnostics["parameters"]),
        "mean": bool(np.all(mean_error < CRITERIA["absolute_mean_error_less_than_raw_mean_mcse_multiple"] * mcse)),
        "covariance": bool(np.max(covariance_error) < CRITERIA["maximum_standardized_covariance_absolute_error"]),
        "quantiles": bool(np.max(quantile_error) < CRITERIA["maximum_quantile_absolute_error_in_marginal_sd"]),
        "divergences": sum(c["divergences"] for c in chain_summaries) == CRITERIA["allowed_divergences"],
    }
    return {
        "model": name, "algorithm": algorithm, "status": "passed" if all(gate.values()) else "failed",
        "criteria_results": gate, "model_specification": spec,
        "settings": {"seeds": list(seeds), "chains": len(seeds), "initial_positions": [[0.0] * model.dimension] * len(seeds),
                     "burn_in": burn, "retained_draws_per_chain": draws, "step_size": step_size,
                     "hmc_integration_steps": steps if algorithm == "hmc" else None,
                     "nuts_max_depth": 7 if algorithm == "nuts" else None,
                     "scales": scales.tolist(), "adaptation": "off; fixed kernel distributional validation",
                     "scaling_source": "independent analytic posterior marginal SD" if name == "regression" else "identity"},
        "dimension": model.dimension,
        "observations": len(spec["metadata"]["config"]["x"]) if name == "regression" else 10 if name == "beta" else 0,
        "elapsed_seconds": time.perf_counter() - before,
        "gradient_evaluations": sum(c["gradient_evaluations"] for c in chain_summaries),
        "integration_steps": sum(c["integration_steps"] for c in chain_summaries),
        "completed_retained_draws": int(values.shape[0] * values.shape[1]),
        "oracle": {"mean": exact_mean.tolist(), "covariance": covariance.tolist(),
                   "quantiles": exact_quantiles.tolist(),
                   "source": "Beta conjugacy" if name == "beta" else "Gaussian conjugate linear solve" if name == "regression" else "declared Gaussian target"},
        "estimate": {"mean": observed_mean.tolist(), "covariance": observed_covariance.tolist(),
                     "quantiles": observed_quantiles.tolist(), "absolute_mean_error": mean_error.tolist(),
                     "mean_error_in_mcse": (mean_error / mcse).tolist(),
                     "max_standardized_covariance_error": float(np.max(covariance_error)),
                     "max_quantile_error_in_marginal_sd": float(np.max(quantile_error))},
        "diagnostics": diagnostics, "chain_summaries": chain_summaries,
        "raw_model_draws": all_draws, "raw_unconstrained_draws": all_z,
    }


def pilot():
    """Small measured pilot before full cases; primitive work, not a quality claim."""
    model = compile_model(preset("beta")["spec"])
    rng, q, gradients = np.random.default_rng(991), np.zeros(1), 0
    started = time.perf_counter()
    for _ in range(50):
        q, meta = nuts_transition(model, q, rng, 0.2)
        gradients += meta["gradient_evaluations"]
    return {"model": "beta", "algorithm": "nuts", "dimension": 1, "observations": 10,
            "seed": 991, "transitions": 50, "step_size": 0.2,
            "gradient_evaluations": gradients, "elapsed_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "sampler-validation.json")
    args = parser.parse_args()
    report = {
        "format": "posterior-sampler-validation", "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(), "environment": environment(),
        "criteria_declared_before_run": CRITERIA,
        "notes": ["Fixed seeds; no retries or threshold relaxation.",
                  "Rank R-hat/ESS describe recorded chains and cannot exclude missing modes.",
                  "These cases validate fixed kernels through real graph targets; windowed adaptation has separate deterministic tests.",
                  "Timings measure native CPU execution on the recorded environment, not Pyodide or a promised runtime.",
                  "Full raw retained draws include every rejection; samples are never thinned."],
        "pilot": pilot(), "cases": [],
    }
    print("pilot", report["pilot"], flush=True)
    for name in ("beta", "gaussian", "regression"):
        for algorithm in ("hmc", "nuts"):
            result = collect(name, algorithm)
            report["cases"].append(result)
            print(f"{name} {algorithm}: {result['status']}; {result['elapsed_seconds']:.3f}s; "
                  f"{result['gradient_evaluations']} gradients; mean {result['estimate']['mean']}", flush=True)
            # Persist each finished case, including a failure, before continuing.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    failure = collect("beta", "hmc", seeds=[17, 29, 53], burn=100, draws=600,
                      step_size=0.35, steps=10)
    failure["purpose"] = "Observed fixed-length HMC resonance, retained without changing its seeds or threshold."
    failure["original_mean_error_tolerance"] = 0.025
    failure["original_mean_check_passed"] = abs(failure["estimate"]["mean"][0] - 9 / 14) < 0.025
    failure["explanation"] = ("The integration time 10*0.35=3.5 is close to the period implied by the Beta logit target's local curvature. "
                              "High acceptance can coexist with poor mixing. The main validation uses five HMC steps at 0.2; its thresholds were not relaxed.")
    report["observed_failure_cases"] = [failure]
    report["all_required_cases_passed"] = all(c["status"] == "passed" for c in report["cases"])
    args.output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"HMC resonance: mean {failure['estimate']['mean'][0]:.6f}, original mean gate passed={failure['original_mean_check_passed']}")
    print(f"Saved {args.output} ({args.output.stat().st_size:,} bytes)")
    return 0 if report["all_required_cases_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
