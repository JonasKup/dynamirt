"""NumPy simulation and model setup for static 1PL–4PL recovery tests."""

import os

import numpy as np
import numpyro.distributions as dist

from dynamirt import Full, dynamirt
from .helpers import N_ITEMS, N_RESPONDENTS, SEED, fit_responses, save_estimates


def simulate(intercept, discrimination, rng, lower=0.0, upper=1.0):
    """Independent NumPy simulation: p = lower + (upper-lower) * sigmoid(eta)."""
    theta = rng.normal(size=N_RESPONDENTS)
    eta = intercept + theta[:, None] * discrimination
    probability = np.exp(-np.logaddexp(0.0, -eta))
    probability = lower + (upper - lower) * probability
    return rng.binomial(1, probability).astype(float), theta


def fit_simulated(model_type, replication):
    # Separate simulation/fitting seeds; adding replications preserves earlier ones.
    # IDs 3–5 belong to the polytomous models; preserve all existing seeds.
    model_id = {"1PL": 1, "2PL": 2, "3PL": 6, "4PL": 7}[model_type]
    seeds = np.random.SeedSequence([SEED, model_id, replication]).spawn(2)
    simulation_seed, fitting_seed = [int(s.generate_state(1)[0]) for s in seeds]
    intercept = np.linspace(-1.5, 1.5, N_ITEMS)
    discrimination = np.ones(N_ITEMS)
    if model_type != "1PL":
        discrimination = np.random.default_rng(314159).permutation(np.linspace(0.6, 1.6, N_ITEMS))
    lower = np.linspace(0.05, 0.30, N_ITEMS) if model_type in ("3PL", "4PL") else 0.0
    upper = np.linspace(0.95, 0.70, N_ITEMS) if model_type == "4PL" else 1.0
    responses, theta = simulate(intercept, discrimination, np.random.default_rng(simulation_seed),
                                lower=lower, upper=upper)

    # Standard-normal traits fix location/scale; positive slopes fix polarity.
    kwargs = {} if model_type == "1PL" else {"loadings": Full(dist.LogNormal(0, 0.5))}
    model = dynamirt(model_type=model_type, n_latent=1, corr=False, **kwargs)
    raw, divergences = fit_responses(model, responses, fitting_seed)
    # These raw sites equal the parameters for the unpooled, iid specification.
    truth = {"intercept": intercept, "theta": theta}
    draws = {"intercept": np.asarray(raw["item_intercept_raw"])[:, 0, :, 0],
             "theta": np.asarray(raw["residuals_raw"])[:, :, 0, 0]}
    if model_type != "1PL":
        truth["discrimination"] = discrimination
        draws["discrimination"] = np.asarray(raw["loadings.full"])[:, :, 0]
    if model_type in ("3PL", "4PL"):
        truth["lower_asymptote"] = lower
        draws["lower_asymptote"] = np.asarray(raw["lower_asymptote"])
    if model_type == "4PL":
        truth["upper_asymptote"] = upper
        # The sampled gap is a fraction of the remaining probability range.
        lower_draws = draws["lower_asymptote"]
        draws["upper_asymptote"] = lower_draws + (1 - lower_draws) * np.asarray(raw["asymptote_gap"])
    if os.environ.get("DYNAMIRT_RECOVERY_SAVE") == "1":
        save_estimates(model_type, replication, truth, draws)
    return truth, draws, divergences
