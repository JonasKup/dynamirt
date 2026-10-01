"""Shared fitting, diagnostics, and saving for tests and the standalone study."""

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from dynamirt import fit_mcmc

from .config import SMALL, SEED

# Pytest runs one small replication per family.
REPLICATIONS = 1


def diagnostics_for(draws):
    """Extrema across parameters; axes 0 and 1 must be chain and draw."""
    import arviz as az

    values = {"max_rhat": [], "min_bulk_ess": [], "min_tail_ess": []}
    for array in draws.values():
        array = np.asarray(array)
        if array.shape[0] > 1:
            values["max_rhat"].extend(np.asarray(az.rhat(array)).ravel())
        values["min_bulk_ess"].extend(np.asarray(az.ess(array, method="bulk")).ravel())
        values["min_tail_ess"].extend(np.asarray(
            az.ess(array, method="tail", prob=(0.025, 0.975))).ravel())
    result = {}
    for name, entries in values.items():
        result[name] = (float(max(entries) if name == "max_rhat" else min(entries))
                        if entries and np.isfinite(entries).all() else None)
    result["diagnostics_finite"] = all(
        value is not None for name, value in result.items()
        if name != "max_rhat" or next(iter(draws.values())).shape[0] > 1)
    return result


def fit_responses(model, responses, fitting_seed, settings=SMALL, diagnostics=False):
    """Shared NUTS fit; diagnostics are computed only for saved study runs."""
    jax.clear_caches()  # Release previous fits' compiled code on small machines.
    started = time.perf_counter()
    fit = fit_mcmc(
        model, responses, {}, rng_key=jax.random.PRNGKey(fitting_seed),
        return_deterministic=False,
        kernel_kwargs={"target_accept_prob": 0.9, "max_tree_depth": settings.max_tree_depth},
        mcmc_kwargs={"num_warmup": settings.warmup, "num_samples": settings.samples, "num_chains": settings.chains,
                     "chain_method": settings.chain_method, "progress_bar": False},
    )
    mcmc = fit.inference
    raw = mcmc.get_samples()
    divergences = int(np.sum(mcmc.get_extra_fields()["diverging"]))
    fit_seconds = time.perf_counter() - started
    metadata = {}
    if diagnostics:
        metadata = {
            "raw_diagnostics": diagnostics_for(mcmc.get_samples(group_by_chain=True)),
            "divergences": divergences, "fit_seconds": fit_seconds,
            "fitting_seed": fitting_seed,
        }
    return raw, divergences, metadata


def rmse(draws, truth):
    return float(np.sqrt(np.mean((draws.mean(axis=0) - truth) ** 2)))


def save_estimates(model_type, replication, truth, draws, metadata, simulation_seed,
                   output, settings=SMALL, valid=None):
    """Item summaries only; equal-tailed 95% intervals, intercepts (not difficulties)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"{model_type}_{replication:03d}"
    item_draws = {}
    for name, values in draws.items():
        if name == "theta":
            continue
        if truth[name].ndim == 2 and valid is not None:
            values = values[:, valid]
        item_draws[name] = values.reshape(settings.chains, settings.samples, -1)
    metadata = {
        **metadata, "model": model_type, "replication": replication,
        "simulation_seed": simulation_seed,
        "item_diagnostics": diagnostics_for(item_draws),
        "settings": {**asdict(settings), "target_accept_prob": 0.9,
                     "seed": SEED,
                     "x64": bool(jax.config.x64_enabled)},
    }
    with stem.with_suffix(".csv.tmp").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "replication", "parameter", "item", "threshold", "truth",
                         "posterior_mean", "posterior_sd", "lower_95", "upper_95"])
        for name, values in draws.items():
            if name == "theta":
                continue
            lower, upper = np.quantile(values, [0.025, 0.975], axis=0)
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            for index in np.ndindex(truth[name].shape):
                if len(index) == 2 and valid is not None and not valid[index]:
                    continue
                writer.writerow([model_type, replication, name, index[0],
                                 index[1] if len(index) == 2 else "", truth[name][index],
                                 mean[index], sd[index], lower[index], upper[index]])
    stem.with_suffix(".csv.tmp").replace(stem.with_suffix(".csv"))
    # JSON is written last: a complete result requires both files.
    stem.with_suffix(".json.tmp").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    stem.with_suffix(".json.tmp").replace(stem.with_suffix(".json"))
