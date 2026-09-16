"""Small static recovery checks: python -m pytest tests/recovery -q.

Set DYNAMIRT_RECOVERY_REPS=100 for more replications (default: 1).
Set DYNAMIRT_RECOVERY_SAVE=1 to save one compact CSV per fit in output/ here.
Adjust the constants below for larger datasets or longer fits. The short,
single-chain defaults are regression checks, not paper-quality inference.
Item truth stays fixed across replications; traits and responses are redrawn.
"""

import csv
import os
from pathlib import Path

import jax
import numpy as np

from dynamirt import fit_mcmc

REPLICATIONS = int(os.environ.get("DYNAMIRT_RECOVERY_REPS", "1"))
N_RESPONDENTS, N_ITEMS = 300, 12
WARMUP, SAMPLES, CHAINS = 200, 200, 1
SEED = 0


def fit_responses(model, responses, fitting_seed):
    """Shared small NUTS fit for binary and ordinal recovery tests."""
    jax.clear_caches()  # Release previous fits' compiled code on small machines.
    fit = fit_mcmc(
        model, responses, {}, rng_key=jax.random.PRNGKey(fitting_seed),
        return_deterministic=False,
        kernel_kwargs={"target_accept_prob": 0.9, "max_tree_depth": 8},
        mcmc_kwargs={"num_warmup": WARMUP, "num_samples": SAMPLES, "num_chains": CHAINS,
                     "chain_method": "sequential", "progress_bar": False},
    )
    mcmc = fit.inference
    return mcmc.get_samples(), int(np.sum(mcmc.get_extra_fields()["diverging"]))


def rmse(draws, truth):
    return float(np.sqrt(np.mean((draws.mean(axis=0) - truth) ** 2)))


def save_estimates(model_type, replication, truth, draws, valid=None):
    """Item summaries only; equal-tailed 95% intervals, intercepts (not difficulties)."""
    output = Path(__file__).parent / "output"
    output.mkdir(exist_ok=True)
    with (output / f"{model_type}_{replication:03d}.csv").open("w", newline="") as handle:
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
