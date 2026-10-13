"""Float64 multi-chain diagnostics, independently implemented from their definitions.

Definitions follow Vehtari et al. (2021); numerical fixtures use ArviZ 0.21.0.
Production code imports no inference or diagnostics package. See docs/diagnostics.md.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len
from scipy.special import ndtri
from scipy.stats import rankdata


DEFINITIONS = {
    "reference": "ArviZ 0.21.0 public rhat, ess and mcse functions; Vehtari et al. (2021)",
    "shape": "chain, draw, parameter; retained post-warmup draws including rejections",
    "split": "First and last floor(draws/2) draws; omit the middle draw when odd",
    "rank_rhat": "Split R-hat of pooled average ranks transformed by Phi^-1((rank-3/8)/(N+1/4))",
    "folded_rhat": "Rank-normalized split R-hat of absolute deviations from the pooled split median",
    "rhat": "Maximum of rank_rhat and folded_rhat",
    "ess_bulk": "Geyer initial positive/monotone sequence ESS of rank-normalized split chains",
    "ess_tail": "Minimum split indicator ESS at pooled raw 5th and 95th percentiles",
    "ess_mean": "Geyer initial positive/monotone sequence ESS of raw-scale split chains",
    "mcse_mean": "Pooled raw sample standard deviation divided by sqrt(raw-scale ess_mean)",
    "ess_bound": "Integrated autocorrelation lower bound 1/log10(N); ESS may exceed N",
    "unavailable": "Nonfinite, any constant original chain, fewer than 2 chains or 4 draws",
    "status": "R-hat >1.01: chains disagree; ESS <100 per chain or <50 draws: preliminary; no convergence proof",
}


def split_chains(values: np.ndarray) -> np.ndarray:
    """Split chains, discarding only the middle observation of odd length."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError("Splitting requires [chain, draw] and at least four draws")
    half = values.shape[1] // 2
    return np.concatenate((values[:, :half], values[:, -half:]), axis=0)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Pooled average ranks with Blom's normal-score transformation."""
    values = np.asarray(values, dtype=np.float64)
    ranks = rankdata(values.ravel(), method="average").reshape(values.shape)
    return ndtri((ranks - 0.375) / (values.size + 0.25))


def _rhat(values: np.ndarray) -> float:
    _, draws = values.shape
    within = np.var(values, axis=1, ddof=1).mean()
    between_per_draw = np.var(values.mean(axis=1), ddof=1)
    if not np.isfinite(within) or within <= 0:
        return float("nan")
    return float(np.sqrt((draws - 1) / draws + between_per_draw / within))


def _autocovariance(values: np.ndarray) -> np.ndarray:
    """Biased (denominator N) autocovariance, computed by zero-padded FFT."""
    draws = values.shape[1]
    centered = values - values.mean(axis=1, keepdims=True)
    fft_size = next_fast_len(2 * draws)
    spectrum = np.fft.rfft(centered, n=fft_size, axis=1)
    return np.fft.irfft(spectrum * spectrum.conj(), n=fft_size, axis=1)[:, :draws] / draws


def _ess(values: np.ndarray) -> float:
    """Multi-chain Geyer ESS with finite-sample antithetic correction.

    The final unpaired positive even lag and log10(N) upper ESS bound match
    the pinned reference's finite-sample convention. Constant transformed
    indicators are deliberately undefined rather than assigned perfect ESS.
    """
    values = np.asarray(values, dtype=np.float64)
    chains, draws = values.shape
    if np.ptp(values) == 0:
        return float("nan")
    covariances = _autocovariance(values).mean(axis=0)
    within = covariances[0] * draws / (draws - 1)
    variance = covariances[0] + np.var(values.mean(axis=1), ddof=1)
    if variance <= 0 or not np.isfinite(variance):
        return float("nan")
    correlations = 1.0 - (within - covariances) / variance
    correlations[0] = 1.0
    retained = np.zeros(draws, dtype=np.float64)
    retained[:2] = correlations[:2]
    even, odd, lag = correlations[0], correlations[1], 1
    while lag < draws - 3 and even + odd > 0:
        even, odd = correlations[lag + 1:lag + 3]
        if even + odd >= 0:
            retained[lag + 1:lag + 3] = (even, odd)
        lag += 2
    final_pair = lag - 2
    if even > 0:
        retained[final_pair + 1] = even
    for lag in range(1, final_pair - 1, 2):
        previous = retained[lag - 1:lag + 1].sum()
        current = retained[lag + 1:lag + 3].sum()
        if current > previous:
            retained[lag + 1:lag + 3] = previous / 2
    tau = -1 + 2 * retained[:final_pair + 1].sum() + retained[final_pair + 1:final_pair + 2].sum()
    count = chains * draws
    return float(count / max(tau, 1 / np.log10(count)))


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _parameter(values: np.ndarray, name: str) -> dict:
    row = dict.fromkeys(("mean", "sd", "q05", "q50", "q95", "rhat", "rank_rhat",
                         "folded_rhat", "ess_bulk", "ess_tail", "ess_mean", "mcse_mean"))
    row.update(name=name, status="unavailable", message="too few draws to assess")
    chains, draws = values.shape
    if not np.isfinite(values).all():
        row["message"] = "nonfinite draws; diagnostics unavailable"
        return row
    if values.size:
        row["mean"] = _finite(values.mean())
        row["sd"] = _finite(values.std(ddof=1)) if values.size > 1 else None
        row.update(zip(("q05", "q50", "q95"), map(float, np.quantile(values, [0.05, 0.5, 0.95]))))
    if chains < 2 or draws < 4:
        return row
    if np.any(np.ptp(values, axis=1) == 0):
        row["message"] = "constant chain; diagnostics unavailable"
        return row
    split = split_chains(values)
    normalized = rank_normalize(split)
    rank_rhat = _rhat(normalized)
    folded_rhat = _rhat(rank_normalize(abs(split - np.median(split))))
    raw_ess = _ess(split)
    quantiles = np.quantile(values, [0.05, 0.95])
    tail_ess = [_ess(split_chains(values <= quantile)) for quantile in quantiles]
    row.update(rank_rhat=_finite(rank_rhat), folded_rhat=_finite(folded_rhat),
               rhat=_finite(np.maximum(rank_rhat, folded_rhat)),
               ess_bulk=_finite(_ess(normalized)), ess_tail=_finite(np.minimum(*tail_ess)),
               ess_mean=_finite(raw_ess), mcse_mean=_finite(values.std(ddof=1) / np.sqrt(raw_ess)))
    if any(row[key] is None for key in ("rhat", "ess_bulk", "ess_tail", "mcse_mean")):
        row.update(status="unreliable", message="a diagnostic is undefined; inspect chains and ties")
    elif row["rhat"] > 1.01:
        row.update(status="unreliable", message="chains disagree")
    elif draws < 50 or min(row["ess_bulk"], row["ess_tail"]) < 100 * chains:
        row.update(status="preliminary", message="limited effective draws; estimates are preliminary")
    else:
        row.update(status="assessed", message="no disagreement detected; unseen modes remain possible")
    return row


def diagnose(samples: np.ndarray, names: list[str] | None = None) -> dict:
    """Return strict-JSON-safe retained-MCMC summaries and diagnostic statuses.

    This function is for MCMC chains, never variational draws. It makes no
    inference from transition acceptance rates and never labels a run converged.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[2] < 1:
        raise ValueError("samples must have shape [chain, draw, parameter]")
    if values.shape[0] > 4 or values.shape[1] > 2000 or values.shape[2] > 128:
        raise ValueError("diagnostic limits: 4 chains, 2000 draws and 128 reported quantities")
    if names is None:
        names = [f"parameter_{index}" for index in range(values.shape[2])]
    if len(names) != values.shape[2] or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("names must be distinct strings, one for each parameter")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rows = [_parameter(values[:, :, index], name) for index, name in enumerate(names)]
    priority = {"assessed": 0, "preliminary": 1, "unreliable": 2, "unavailable": 3}
    worst = max(rows, key=lambda row: priority[row["status"]])
    return {"status": worst["status"], "message": worst["message"],
            "shape": list(values.shape), "parameters": rows, "definitions": dict(DEFINITIONS)}
