"""NumPy simulation and model setup for static 1PL/2PL recovery tests."""

import os

import numpy as np
import numpyro.distributions as dist

from dynamirt import Full, dynamirt
from .helpers import N_ITEMS, N_RESPONDENTS, SEED, fit_responses, save_estimates


def simulate(intercept, discrimination, rng):
    """Independent NumPy simulation: logit(p) = intercept + discrimination * theta."""
    theta = rng.normal(size=N_RESPONDENTS)
    eta = intercept + theta[:, None] * discrimination
    probability = np.exp(-np.logaddexp(0.0, -eta))
    return rng.binomial(1, probability).astype(float), theta


def fit_simulated(model_type, replication):
    # Separate simulation/fitting seeds; adding replications preserves earlier ones.
    model_id = {"1PL": 1, "2PL": 2}[model_type]
    seeds = np.random.SeedSequence([SEED, model_id, replication]).spawn(2)
    simulation_seed, fitting_seed = [int(s.generate_state(1)[0]) for s in seeds]
    intercept = np.linspace(-1.5, 1.5, N_ITEMS)
    discrimination = np.ones(N_ITEMS)
    if model_type == "2PL":
        discrimination = np.random.default_rng(314159).permutation(np.linspace(0.6, 1.6, N_ITEMS))
    responses, theta = simulate(intercept, discrimination, np.random.default_rng(simulation_seed))

    # Standard-normal traits fix location/scale; positive 2PL slopes fix polarity.
    kwargs = {} if model_type == "1PL" else {"loadings": Full(dist.LogNormal(0, 0.5))}
    model = dynamirt(model_type=model_type, n_latent=1, corr=False, **kwargs)
    raw, divergences = fit_responses(model, responses, fitting_seed)
    # These raw sites equal the parameters for the unpooled, iid specification.
    truth = {"intercept": intercept, "theta": theta}
    draws = {"intercept": np.asarray(raw["item_intercept_raw"])[:, 0, :, 0],
             "theta": np.asarray(raw["residuals_raw"])[:, :, 0, 0]}
    if model_type == "2PL":
        truth["discrimination"] = discrimination
        draws["discrimination"] = np.asarray(raw["loadings.full"])[:, :, 0]
    if os.environ.get("DYNAMIRT_RECOVERY_SAVE") == "1":
        save_estimates(model_type, replication, truth, draws)
    return truth, draws, divergences
