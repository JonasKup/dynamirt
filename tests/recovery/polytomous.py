"""NumPy simulation for GRM and PCM/GPCM, with 2–5 categories between items."""

import os

import numpy as np
import numpyro.distributions as dist

from dynamirt import Full, dynamirt
from .helpers import N_ITEMS, N_RESPONDENTS, SEED, fit_responses, save_estimates

N_CATEGORIES = np.resize([2, 3, 4, 5], N_ITEMS)
VALID = np.arange(N_CATEGORIES.max() - 1) < N_CATEGORIES[:, None] - 1


def simulate(model_type, rng):
    """Generate probabilities directly, independently of the package likelihood."""
    theta = rng.normal(size=N_RESPONDENTS)
    discrimination = (np.ones(N_ITEMS) if model_type == "PCM" else
                      np.random.default_rng(314159).permutation(np.linspace(0.6, 1.6, N_ITEMS)))
    thresholds = np.zeros(VALID.shape)
    responses = np.empty((N_RESPONDENTS, N_ITEMS))
    for item, count in enumerate(N_CATEGORIES):
        levels = np.linspace(-0.6 * (count - 2), 0.6 * (count - 2), count - 1)
        # PCM/GPCM steps need not be ordered; exercise that as well.
        if model_type != "GRM" and item % 2:
            levels = levels[::-1]
        levels = levels + np.linspace(-0.5, 0.5, N_ITEMS)[item]
        thresholds[item, :count - 1] = levels
        eta = discrimination[item] * theta[:, None]
        if model_type == "GRM":
            cumulative = np.exp(-np.logaddexp(0.0, eta - levels))
            probability = np.diff(np.column_stack([np.zeros(N_RESPONDENTS), cumulative,
                                                   np.ones(N_RESPONDENTS)]), axis=1)
        else:
            logits = np.column_stack([np.zeros(N_RESPONDENTS), np.cumsum(eta - levels, axis=1)])
            probability = np.exp(logits - logits.max(axis=1, keepdims=True))
            probability /= probability.sum(axis=1, keepdims=True)
        assert np.all(probability >= 0)
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
        responses[:, item] = (rng.random((N_RESPONDENTS, 1)) >
                              probability.cumsum(axis=1)[:, :-1]).sum(axis=1)
    name = "cutpoints" if model_type == "GRM" else "steps"
    truth = {name: thresholds, "theta": theta}
    if model_type != "PCM":
        truth["discrimination"] = discrimination
    return responses, truth


def fit_simulated(model_type, replication):
    model_id = {"GRM": 3, "PCM": 4, "GPCM": 5}[model_type]
    seeds = np.random.SeedSequence([SEED, model_id, replication]).spawn(2)
    simulation_seed, fitting_seed = [int(s.generate_state(1)[0]) for s in seeds]
    responses, truth = simulate(model_type, np.random.default_rng(simulation_seed))
    assert ((responses >= 0) & (responses < N_CATEGORIES)).all()
    kwargs = {} if model_type == "PCM" else {"loadings": Full(dist.LogNormal(0, 0.5))}
    model = dynamirt(model_type=model_type, n_latent=1, corr=False,
                     model_type_kwargs={"n_cat": N_CATEGORIES}, **kwargs)
    raw, divergences = fit_responses(model, responses, fitting_seed)
    name = "cutpoints" if model_type == "GRM" else "steps"
    compact = np.asarray(raw[name + "_raw"])
    thresholds = np.zeros((len(compact), *VALID.shape))
    thresholds[:, VALID] = compact
    if model_type == "GRM":
        # First raw value is the location; remaining raw values are log gaps.
        thresholds[:, :, 1:] = np.exp(thresholds[:, :, 1:])
        thresholds = np.where(VALID, thresholds.cumsum(axis=-1), 0.0)
    draws = {name: thresholds, "theta": np.asarray(raw["residuals_raw"])[:, :, 0, 0]}
    if model_type != "PCM":
        draws["discrimination"] = np.asarray(raw["loadings.full"])[:, :, 0]
    if os.environ.get("DYNAMIRT_RECOVERY_SAVE") == "1":
        save_estimates(model_type, replication, truth, draws, valid=VALID)
    return truth, draws, divergences
