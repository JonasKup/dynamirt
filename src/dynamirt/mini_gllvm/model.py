from typing import Mapping, List, Callable

import numpyro
import numpy as np

import jax.numpy as jnp
from jax.typing import ArrayLike

from ._context import _Context
from .loadings import Unconstrained
    
def gllvm(
    responses: None | ArrayLike,
    covariates: Mapping[str, ArrayLike],
    family: Callable,
    n_latent: int=1,
    full_rank_regression: List[Callable]=None,
    latent_regression: List[Callable]=None,
    loadings: Callable | None =None,
    n_obs: int=None, # for predictive
    n_var: int=None,
    latent_site_name: str="u"
    ):
    
    """Generalized linear latent variable model.

    Regresses `n_var` responses jointly on a full-rank linear predictor plus a
    reduced-rank (`n_latent`) term, `mu = eta + u @ loadings`. NaNs in
    `responses` are masked out of the likelihood.

    Args:
        responses: (n_obs, n_var) observations; NaN marks missing.
        covariates: name -> array of length n_obs (or broadcastable). Keys
            "one_" (ones, for intercepts) and "row_" (row index) are added
            automatically.
        n_latent: dimensionality of the latent space.
        full_rank_regression: terms called as `term(ctx, n_var) -> (n_obs, n_var)`,
            summed into the full-rank predictor.
        latent_regression: terms called as `term(ctx, n_latent) -> (n_obs, n_latent)`,
            summed into the latent scores.
        family: called as `family(mu, ctx) -> dist.Distribution` over responses.
    """
    if responses is None and (n_obs is None or n_var is None):
        raise ValueError("n_obs and n_var must be specified explicitly if no responses are given.")
    
    # loadings default to unconstrained
    loadings = Unconstrained() if loadings is None else loadings

    # n_obs: total number of obs. Doesn't necessarily need to be number of sites/respondents in repeated measurement scenarios
    # n_var: number of variables to simultanously regress on
    if responses is not None:
        n_obs, n_var = responses.shape 
        
    covariates = {**covariates,
                  "one_": np.array([1.0]),  # for intercepts
                  "row_": np.arange(n_obs)} # stand-in for ID column in scenarios w/o repeated measures where n_obs == n_site/respondent
        
    ctx = _Context(responses, covariates, n_obs, n_var, n_latent) # context to pass to subfunctions like term and family
    
    # full-rank regression (DIF under IRT)
    eta = jnp.zeros((n_obs, n_var))
    if full_rank_regression is not None:
        for term in full_rank_regression:
            eta += term(ctx, n_var) # (n_obs, n_var)
    
    # reduced-rank regression
    u = jnp.zeros((n_obs, n_latent))
    if latent_regression is not None:
        for term in latent_regression:
            u += term(ctx, n_latent) # (n_obs, n_latent)
    
    loadings_matrix = loadings(ctx) # (n_var, n_latent)
        
    latent_contributions = u @ loadings_matrix.T # (n_obs, n_var)
    
    numpyro.deterministic(latent_site_name, u)
    numpyro.deterministic("loadings", loadings_matrix)
    
    mu = eta + latent_contributions
    
    # likelihood based on user supplied family function
    # responses should always be float to carry nan
    # must be manually cast to int if the distribution returned by family_fn only has discrete support (e.g., ordinal models)
    family_fn = family(mu, ctx)
    
    if responses is None:
        numpyro.sample("Y", family_fn)
    else:
        mask = jnp.isnan(responses)
        obs=jnp.where(mask, 0.0, responses)
        
        if family_fn.support.is_discrete:
            obs = obs.astype(jnp.result_type(int))
        numpyro.sample("Y", family_fn.mask(~mask), obs=obs)