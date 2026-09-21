from typing import Mapping, List, Callable

import numpyro
import numpy as np

import jax.numpy as jnp
from jax.typing import ArrayLike

from ._context import _Context
from .loadings import Full


def _validate_responses(responses, discrete, n_cat=None):
    values = np.asarray(responses)
    observed = ~np.isnan(values)
    if not np.all(np.isfinite(values[observed])):
        raise ValueError("Responses must be finite or NaN")
    if discrete and np.any(values[observed] != np.floor(values[observed])):
        raise ValueError("Responses must be integers or NaN")
    if n_cat is not None and np.any(observed & ((values < 0) | (values >= n_cat))):
        raise ValueError("Response category out of range")


def gllvm(
    responses: None | ArrayLike,
    covariates: Mapping[str, ArrayLike],
    family: Callable,
    n_latent: int=1,
    full_rank_regression: List[Callable]=None,
    latent_regression: List[Callable]=None,
    loadings: Callable | None =None,
    n_obs: int=None,
    n_var: int=None,
    train_covariates: Mapping[str, ArrayLike]=None,
    train_obs: int=None,
    latent_site_name: str="u"
    ):
    
    """Generalized linear latent variable model.

    Regresses `n_var` responses jointly on a full-rank linear predictor plus a
    reduced-rank (`n_latent`) term:
    
    mu = eta + u @ loadings.T
    
    where eta is the sum of full-rank regression terms, u collects
    the latent scores, and loadings is a (n_var, n_latent) matrix.
    NaN entries in `responses` are masked out of the likelihood.
    
    Args:
        responses: Observation matrix of shape (n_obs, n_var). NaN marks
            missing data. Pass None for prior-predictive sampling, in
            which case n_obs and n_var must be given explicitly.
        covariates: Mapping from covariate name to array of length n_obs
            (or broadcastable scalar). Two keys are added automatically:
            ``"one_"`` (constant 1, for intercepts) and ``"row_"``
            (observation index, a stand-in for a respondent ID when
            n_obs equals the number of respondents).
        family: Called as ``family(mu, ctx)`` and must return a
            ``numpyro.distributions.Distribution`` over responses.
        n_latent: Dimensionality of the latent space. Defaults to 1.
        full_rank_regression: List of term callables, each with signature
            ``term(ctx, n_var) -> (n_obs, n_var)``. Their outputs are
            summed into the full-rank predictor eta.
        latent_regression: List of term callables, each with signature
            ``term(ctx, n_latent) -> (n_obs, n_latent)``. Their outputs
            are summed into the latent scores u.
        loadings: A loading-matrix factory ``loadings(ctx) ->
            (n_var, n_latent)``. Defaults to ``Full()`` (unconstrained).
        n_obs: Number of observations. Required when responses is None.
        n_var: Number of response variables. Required when responses is
            None.
        train_covariates: Covariates used for fitting the model.
            Only required when using Predictive with ExactGPs.
        train_obs: Number of observations when fitting the model.
            Only required when using Predictive with ExactGPs.
        latent_site_name: Name of the NumPyro deterministic site that
            stores the latent scores u. Defaults to ``"u"``.
    """
    
    if responses is None and (n_obs is None or n_var is None):
        raise ValueError("n_obs and n_var must be specified explicitly if no responses are given.")
    
    # loadings default to unconstrained
    loadings = Full() if loadings is None else loadings

    # n_obs: total number of obs. Doesn't necessarily need to be number of sites/respondents in repeated measurement scenarios
    # n_var: number of variables to simultanously regress on
    if responses is not None:
        n_obs, n_var = responses.shape 
        
    covariates = {**covariates,
                  "one_": np.array([1.0]),  # for intercepts
                  "row_": np.arange(n_obs)} # stand-in for ID column in scenarios w/o repeated measures where n_obs == n_site/respondent
        
    if train_covariates is not None:
        train_covariates = {
            **train_covariates,
            "one_": np.array([1.0]),
            "row_": np.arange(train_obs),
        }
    
    is_predictive = train_covariates is not None
    
    ctx = _Context(
        responses,
        covariates,
        n_obs, 
        n_var, 
        n_latent,
        is_predictive,
        train_covariates,
        train_obs
        ) # context to pass to subfunctions like term and family
    
    # full-rank regression (DIF under IRT)
    eta = jnp.zeros((n_obs, n_var))
    if full_rank_regression is not None:
        for term in full_rank_regression:
            contribution = term(ctx, n_var) # (n_obs, n_var)
            eta += contribution
            numpyro.deterministic(f"{term.name}_eta", contribution) # makes term need to carry a name field

    
    # reduced-rank regression
    u = jnp.zeros((n_obs, n_latent))
    if latent_regression is not None:
        for term in latent_regression:
            contribution = term(ctx, n_latent) # (n_obs, n_latent)
            u += contribution
            numpyro.deterministic(f"{term.name}_latent", contribution) # makes term need to carry a name field
    
    with numpyro.handlers.scope(prefix="loadings", divider="."):
        loadings_matrix = loadings(ctx) # (n_var, n_latent)
        
    latent_contributions = u @ loadings_matrix.T # (n_obs, n_var)
    
    mu = eta + latent_contributions
    
    numpyro.deterministic(latent_site_name, u)
    numpyro.deterministic("loadings", loadings_matrix)
    
    # likelihood based on user supplied family function
    # responses should always be float to carry nan
    # must be manually cast to int if the distribution returned by family_fn only has discrete support (e.g., ordinal models)
    family_fn = family(mu, ctx)
    
    if responses is None:
        numpyro.sample("Y", family_fn)
    else:
        _validate_responses(responses, family_fn.support.is_discrete, getattr(family, "n_cat", None))
        mask = jnp.isnan(responses)
        obs=jnp.where(mask, 0.0, responses)
        
        if family_fn.support.is_discrete:
            obs = obs.astype(jnp.result_type(int))
        numpyro.sample("Y", family_fn.mask(~mask), obs=obs)