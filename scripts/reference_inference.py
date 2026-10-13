"""CPU-only, development-only inference comparison against emcee 3.1.6.

Run: .venv/bin/python scripts/reference_inference.py
The first pilot is measured before the predeclared evaluation budget. All
predefined seeds are retained, and any gate failure remains in the report.
The ensemble reference uses independently coded normalized target densities;
Posterior executes its ordinary compiled preset graph through Experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/posterior-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/posterior-cache")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import emcee
import numpy as np
import scipy
from scipy import stats
from scipy.special import betaln, expit

from posterior_lab import __version__
from posterior_lab.diagnostics import diagnose
from posterior_lab.examples import preset
from posterior_lab.runtime import Experiment


SETTINGS = {"chains": 4, "warmup": 300, "draws": 1000, "step_size": 0.15,
            "steps": 8, "max_depth": 7, "target_accept": 0.85,
            "max_seconds": 120, "max_gradient_evaluations": 500000}
REFERENCE_SETTINGS = {"walkers": 16, "warmup_steps": 500, "retained_steps": 2000,
                      "move": "emcee.moves.StretchMove(a=2.0)", "threads": 1}
SEEDS = {"gaussian": 307, "beta": 811}
GATES = {"maximum_rhat": 1.03, "minimum_bulk_ess": 200, "minimum_tail_ess": 150,
         "mean_standard_errors": 5, "mean_absolute_fraction_of_true_sd": 0.04,
         "covariance_absolute_fraction_of_true_sd_product": 0.18,
         "quantile_absolute_fraction_of_true_sd": 0.2,
         "reference_minimum_steps_per_autocorrelation_time": 50}


def oracle(name):
    if name == "gaussian":
        covariance = np.array([[1.0, 0.95], [0.95, 1.0]])
        precision = np.linalg.inv(covariance)
        constant = 2 * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1]

        def target(z):
            return float(-0.5 * (constant + z @ precision @ z))

        return {"target": target, "dimension": 2, "mean": np.zeros(2),
                "covariance": covariance, "quantiles": np.tile(stats.norm.ppf([0.05, 0.5, 0.95])[:, None], (1, 2)),
                "report": lambda z: z, "description": "Normalized N([0,0], [[1,.95],[.95,1]])"}

    def target(z):
        # Beta(9,5) after the logit transform: exponents include Jacobian.
        return float(-9 * np.logaddexp(0, -z[0]) - 5 * np.logaddexp(0, z[0]) - betaln(9, 5))

    return {"target": target, "dimension": 1, "mean": np.array([9 / 14]),
            "covariance": np.array([[9 * 5 / (14**2 * 15)]]),
            "quantiles": stats.beta.ppf([0.05, 0.5, 0.95], 9, 5)[:, None],
            "report": expit, "description": "Normalized Beta(9,5) posterior in logit coordinates"}


def summary(samples, truth):
    flat = samples.reshape(-1, samples.shape[-1])
    mean = flat.mean(axis=0)
    covariance = np.atleast_2d(np.cov(flat, rowvar=False, ddof=1))
    quantiles = np.quantile(flat, [0.05, 0.5, 0.95], axis=0)
    sd = np.sqrt(np.diag(truth["covariance"]))
    return {"mean": mean.tolist(), "covariance": covariance.tolist(), "quantiles_05_50_95": quantiles.tolist(),
            "mean_error": (mean - truth["mean"]).tolist(),
            "maximum_scaled_covariance_error": float(np.max(np.abs(covariance - truth["covariance"]) / np.outer(sd, sd))),
            "maximum_scaled_quantile_error": float(np.max(np.abs(quantiles - truth["quantiles"]) / sd))}


def posterior_run(name, algorithm, settings):
    experiment = Experiment(preset(name)["spec"], {**settings, "algorithm": algorithm})
    start = time.perf_counter()
    while experiment.status == "running":
        experiment.step(25)
    wall_seconds = time.perf_counter() - start
    result = experiment.snapshot(checkpoint=False)
    if experiment.status != "complete":
        raise RuntimeError(f"{name}/{algorithm}: {experiment.status}: {experiment.reason}")
    samples = np.asarray(result["samples"], dtype=np.float64)
    return samples, {"status": result["status"], "wall_seconds": wall_seconds,
                     "engine_elapsed_seconds": result["stats"]["elapsed_seconds"],
                     "gradient_evaluations": result["stats"]["gradient_evaluations"],
                     "completed_draws": int(np.prod(samples.shape[:2])),
                     "initial_positions": experiment.initial_positions,
                     "settings": result["settings"], "diagnostics": result["diagnostics"],
                     "transition_counts": result["stats"],
                     "frozen_step_sizes": [item["step_size"] for item in result["adaptation"]],
                     "frozen_scales": [item["scales"] for item in result["adaptation"]]}


def reference_run(name, seed):
    truth = oracle(name)
    count = 0

    def counted_target(z):
        nonlocal count
        count += 1
        return truth["target"](z)

    initialization_rng = np.random.default_rng(seed)
    initial = initialization_rng.normal(0, 0.4, (REFERENCE_SETTINGS["walkers"], truth["dimension"]))
    sampler = emcee.EnsembleSampler(REFERENCE_SETTINGS["walkers"], truth["dimension"], counted_target,
                                   moves=emcee.moves.StretchMove(a=2.0))
    sampler.random_state = np.random.RandomState(seed + 1).get_state()
    start = time.perf_counter()
    final = sampler.run_mcmc(initial, REFERENCE_SETTINGS["warmup_steps"], progress=False)
    sampler.reset()
    sampler.run_mcmc(final, REFERENCE_SETTINGS["retained_steps"], progress=False)
    elapsed = time.perf_counter() - start
    unconstrained = sampler.get_chain()  # [step, walker, parameter], coupled walkers.
    samples = truth["report"](unconstrained)
    # Walker sequences are coupled: do not present their R-hat as independent
    # chains. emcee's dedicated ensemble autocorrelation estimate is recorded.
    tau = emcee.autocorr.integrated_time(samples, quiet=True, tol=50)
    effective = samples.shape[0] * samples.shape[1] / tau
    standard_error = samples.reshape(-1, truth["dimension"]).std(axis=0, ddof=1) / np.sqrt(effective)
    return samples, {"reference": "emcee", "version": emcee.__version__, "settings": REFERENCE_SETTINGS,
                     "initialization_seed": seed, "sampler_randomstate_seed": seed + 1,
                     "initial_positions": initial.tolist(), "wall_seconds": elapsed,
                     "log_density_evaluations": count, "completed_draws": int(np.prod(samples.shape[:2])),
                     "mean_acceptance": float(sampler.acceptance_fraction.mean()),
                     "autocorrelation_time": tau.tolist(), "ess_estimate": effective.tolist(),
                     "mcse_mean_estimate": standard_error.tolist(),
                     "steps_per_autocorrelation_time": (samples.shape[0] / tau).tolist(),
                     "diagnostic_note": "Ensemble walkers are coupled; independent-chain R-hat is not applied. ESS/MCSE use emcee's ensemble autocorrelation estimate."}


def checks(report, truth, mcse, reference=False):
    sd = np.sqrt(np.diag(truth["covariance"]))
    result = {"mean": bool(np.all(np.abs(report["mean_error"]) <= GATES["mean_standard_errors"] * np.asarray(mcse) + GATES["mean_absolute_fraction_of_true_sd"] * sd)),
              "covariance": report["maximum_scaled_covariance_error"] <= GATES["covariance_absolute_fraction_of_true_sd_product"],
              "quantiles": report["maximum_scaled_quantile_error"] <= GATES["quantile_absolute_fraction_of_true_sd"]}
    if reference:
        result["autocorrelation_length"] = min(report["steps_per_autocorrelation_time"]) >= GATES["reference_minimum_steps_per_autocorrelation_time"]
    else:
        rows = report["diagnostics"]["parameters"]
        result["rhat"] = all(row["rhat"] is not None and row["rhat"] <= GATES["maximum_rhat"] for row in rows)
        result["bulk_ess"] = all(row["ess_bulk"] is not None and row["ess_bulk"] >= GATES["minimum_bulk_ess"] for row in rows)
        result["tail_ess"] = all(row["ess_tail"] is not None and row["ess_tail"] >= GATES["minimum_tail_ess"] for row in rows)
        result["divergences"] = report["transition_counts"]["divergences"] == 0
    return result


def principal_checks(name, samples, report):
    if name != "gaussian":
        return
    # Marginal diagnostics can hide poorly explored linear combinations. This
    # independent analytic projection does not depend on estimated covariance.
    basis = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    report["principal_diagnostics"] = diagnose(samples @ basis, ["wide: (x+y)/sqrt(2)", "narrow: (x-y)/sqrt(2)"])
    rows = report["principal_diagnostics"]["parameters"]
    report["checks"]["principal_rhat"] = all(row["rhat"] is not None and row["rhat"] <= GATES["maximum_rhat"] for row in rows)
    report["checks"]["principal_bulk_ess"] = all(row["ess_bulk"] is not None and row["ess_bulk"] >= GATES["minimum_bulk_ess"] for row in rows)
    report["checks"]["principal_tail_ess"] = all(row["ess_tail"] is not None and row["ess_tail"] >= GATES["minimum_tail_ess"] for row in rows)


def main():
    if emcee.__version__ != "3.1.6":
        raise RuntimeError("Independent reference must be pinned to emcee 3.1.6")
    destination = ROOT / "reports"
    destination.mkdir(exist_ok=True)
    arrays, models = {}, {}
    # This flag adds the audited projection and documented follow-up to saved
    # runs without resampling them; all original raw draws remain unchanged.
    reuse = "--reuse-existing" in sys.argv[1:]
    existing = json.loads((destination / "reference-inference.json").read_text()) if reuse else None
    existing_arrays = np.load(destination / "reference-inference-draws.npz", allow_pickle=False) if reuse else None
    print("Measured pilot: gaussian NUTS, 4 chains, 50 warmup + 50 draws", flush=True)
    if reuse:
        pilot = existing["pilot"]
    else:
        _, pilot = posterior_run("gaussian", "nuts", {**SETTINGS, "seed": 97, "warmup": 50, "draws": 50})
    pilot = {key: pilot[key] for key in ("wall_seconds", "gradient_evaluations", "completed_draws", "settings")}
    print(json.dumps(pilot), flush=True)
    for name, seed in SEEDS.items():
        truth = oracle(name)
        models[name] = {"oracle": truth["description"], "mean": truth["mean"].tolist(),
                        "covariance": truth["covariance"].tolist(), "quantiles_05_50_95": truth["quantiles"].tolist()}
        if reuse:
            ref_samples = existing_arrays[f"{name}_emcee_step_walker_parameter"]
            ref_report = existing["models"][name]["emcee"]
        else:
            ref_samples, ref_report = reference_run(name, seed + 10000)
        ref_report.update(summary(ref_samples, truth))
        ref_report["checks"] = checks(ref_report, truth, ref_report["mcse_mean_estimate"], reference=True)
        models[name]["emcee"] = ref_report
        arrays[f"{name}_emcee_step_walker_parameter"] = ref_samples
        print(name, "emcee", ref_report["checks"], flush=True)
        for algorithm in ("hmc", "nuts"):
            if reuse:
                samples = existing_arrays[f"{name}_{algorithm}_chain_draw_parameter"]
                report = existing["models"][name][algorithm]
            else:
                samples, report = posterior_run(name, algorithm, {**SETTINGS, "seed": seed})
            report.update(summary(samples, truth))
            mcse = [row["mcse_mean"] for row in report["diagnostics"]["parameters"]]
            report["checks"] = checks(report, truth, mcse)
            difference = np.asarray(report["mean"]) - np.asarray(ref_report["mean"])
            combined_error = np.sqrt(np.asarray(mcse)**2 + np.asarray(ref_report["mcse_mean_estimate"])**2)
            report["independent_reference_mean_difference"] = difference.tolist()
            report["combined_mcse_mean"] = combined_error.tolist()
            report["checks"]["reference_mean_agreement"] = bool(np.all(np.abs(difference) <= 5 * combined_error + 0.04 * np.sqrt(np.diag(truth["covariance"])) ))
            principal_checks(name, samples, report)
            models[name][algorithm] = report
            arrays[f"{name}_{algorithm}_chain_draw_parameter"] = samples
            print(name, algorithm, report["checks"], flush=True)
    # The original 8-step HMC run is retained above. Changing trajectory length
    # at the same seed investigates its narrow-direction exploration failure.
    print("Documented HMC trajectory-length investigation: same Gaussian seed, 7 integration steps", flush=True)
    samples, investigation = posterior_run("gaussian", "hmc", {**SETTINGS, "seed": SEEDS["gaussian"], "steps": 7})
    truth = oracle("gaussian")
    investigation.update(summary(samples, truth))
    investigation["checks"] = checks(investigation, truth, [row["mcse_mean"] for row in investigation["diagnostics"]["parameters"]])
    principal_checks("gaussian", samples, investigation)
    investigation["reason"] = "The original fixed 8-step HMC run has poor narrow principal-direction diagnostics hidden by individual coordinate diagnostics. This same-seed 7-step follow-up tests sensitivity to fixed integration duration; it does not replace the failed run."
    print("gaussian hmc_steps7", investigation["checks"], flush=True)
    arrays["gaussian_hmc_steps7_chain_draw_parameter"] = samples
    models["gaussian"]["hmc_steps7_investigation"] = investigation
    if existing_arrays is not None:
        existing_arrays.close()
    raw_path = destination / "reference-inference-draws.npz"
    np.savez_compressed(raw_path, **arrays)
    passing = all(all(item[algorithm]["checks"].values()) for item in models.values() for algorithm in ("emcee", "hmc", "nuts"))
    report = {"format": "posterior-native-reference-v1", "status": "passed" if passing else "completed_with_failed_checks",
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "machine": platform.machine(), "numpy": np.__version__, "scipy": scipy.__version__,
                              "posterior": __version__, "emcee": emcee.__version__},
              "execution": "Local CPU, sequential, single process; no GPU or inference service",
              "seeds": SEEDS, "declared_gates": GATES, "pilot": pilot, "models": models,
              "audit_note": "Principal-direction diagnostics were added during audit after discovering disagreement hidden by individual-coordinate diagnostics. Numeric thresholds were not loosened; original samples were retained. The seven-step HMC investigation uses the original seed and is an additional run, not a replacement.",
              "raw_draws": {"file": raw_path.name, "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                            "arrays": {key: list(value.shape) for key, value in arrays.items()},
                            "loading": "numpy.load(path, allow_pickle=False); no thinning"},
              "limitations": ["Different algorithms and random generators are compared distributionally; trajectories are not expected to match.",
                              "emcee ensemble walkers are coupled and are not independent diagnostic chains.",
                              "The reference evaluates hand-coded analytic log densities while Posterior evaluates its compiled graph and gradients; wall times are not a pure sampler-speed comparison.",
                              "The original Gaussian fixed 8-step HMC run fails principal-direction diagnostics even though its individual-coordinate diagnostics look reassuring; its samples and failed checks are retained.",
                              "This is a bounded common-model cross-check, not validation of every supported target.",
                              "Timings are observed on the recorded host and are not portable runtime guarantees."],
              "reference_documentation": "https://emcee.readthedocs.io/en/stable/tutorials/quickstart/"}
    path = destination / "reference-inference.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(path, report["status"], flush=True)
    if not passing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
