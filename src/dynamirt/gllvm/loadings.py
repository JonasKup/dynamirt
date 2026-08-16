from typing import Callable

import numpy as np

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.typing import ArrayLike

from ._context import _Context

def Fixed() -> Callable:
    """
    Loading matrix factory that fixes all loadings to 1 so every item loads equally on every latent factor.
    
    Returns:
        A `loadings(ctx)` function that returns a (n_var, n_latent) array of ones.
    """
    def loadings(ctx: _Context):
        
        return jnp.ones((ctx.n_var, ctx.n_latent))
    
    return loadings

def Confirmatory(
    Q: ArrayLike, 
    positive_anchors: ArrayLike = None, 
    free_prior: dist.Distribution | None = None,
    positive_prior: dist.Distribution | None = None
    ) -> Callable:
    """
    Loading matrix factory for confirmatory factor analysis.

    Args:
        Q: Binary (n_var, n_latent) array. Q[i, j] = 1 means item i is
            allowed to load on factor j; 0 means the loading is fixed at 0.
        positive_anchors: Optional binary array of the same shape as Q.
            Marks a subset of Q's nonzero entries as positive-only anchor loadings.
            If None, all free loadings are unconstrained in sign.
            Defaults to None.
        free_prior: Prior distribution for the unconstrained loadings.
            Defaults to Normal(0, 1) if None.
        positive_prior: Prior distribution for the positive anchor loadings.
            Defaults to LogNormal(0, 0.5) if None.

    Returns:
        A `loadings(ctx)` function that returns the (n_var, n_latent)
        loading matrix
    """
    
    free_prior = dist.Normal(0, 1) if free_prior is None else free_prior
    positive_prior = dist.LogNormal(0, 0.5) if positive_prior is None else positive_prior
    
    rows, cols = np.nonzero(np.asarray(Q))
    pos = (np.zeros(rows.size, bool) if positive_anchors is None
           else np.asarray(positive_anchors)[rows, cols].astype(bool))
    free = ~pos
    n_free, n_pos = int(free.sum()), int(pos.sum())

    def loadings(ctx: _Context):
        discrimination = jnp.zeros((ctx.n_var, ctx.n_latent))

        if n_free:
            l = numpyro.sample("confirmatory_free", free_prior.expand((n_free,)).to_event(1))
            discrimination = discrimination.at[rows[free], cols[free]].set(l)

        if n_pos:
            lp = numpyro.sample("confirmatory_positive", positive_prior.expand((n_pos,)).to_event(1))
            discrimination = discrimination.at[rows[pos], cols[pos]].set(lp)

        return discrimination

    return loadings

def Full(prior: dist.Distribution | None = None) -> Callable:
    """
    Loading matrix factory with all loadings drawn from `prior`. 
    For the default Normal(0, 1) this results in an unconstrained loading matrix and is
    equivalent to `Confirmatory` with Q of all ones and no positive anchors and will lead to
    rotational and signed-permutation invariances in most cases.

    Args:
        prior: Prior distribution for each loading. Defaults to Normal(0, 1).

    Returns:
        A `loadings(ctx)` function that returns a (n_var, n_latent) array
        sampled entrywise from `prior`.
    """
    prior = dist.Normal(0, 1) if prior is None else prior
    def loadings(ctx: _Context):
        return numpyro.sample(
            "full", prior.expand((ctx.n_var, ctx.n_latent)).to_event(2)
        )
        
    return loadings

def _half_cauchy_reparam(name, scale, shape=(), event_dim=0):
    """inv gamma reparametrization for half cauchy under SVI"""
    aux = numpyro.sample(f"{name}_aux", dist.InverseGamma(0.5, 1.0 / scale**2).expand(shape).to_event(event_dim))
    x_sq = numpyro.sample(name, dist.InverseGamma(0.5, 1.0 / aux).to_event(event_dim))
    return jnp.sqrt(x_sq)

def Sparsity(tau0=1.0, slab_scale: float=1.0, slab_df: float=4.0) -> Callable:
    """
    Loading matrix factory with regularized horseshoe sparsity prior.
    
    Applies the regularised horseshoe prior of Piironen & Vehtari (2017)
    to the loading matrix, encouraging most loadings toward zero while
    allowing a sparse subset to remain large.

    Reference:
        Juho Piironen. Aki Vehtari (2017). Electron. J. Statist. 11 (2) 5018 - 5051
        https://doi.org/10.1214/17-EJS1337SI 
    
    Args:
        tau0: Global shrinkage scale. A sensible default follows
            Eq. 3.12: ``p0 / (D - p0) * sigma / sqrt(n)`` where p0 is
            the expected number of non-zero loadings and D is the total
            number of loadings. Defaults to 1.0.
        slab_scale: Typical magnitude of a loading that escapes
            shrinkage. Controls the width of the regularising slab.
            Defaults to 1.0.
        slab_df: Degrees of freedom of the Student-t slab controlling
            how heavy-tailed the non-zero loadings may be. Defaults
            to 4.0.
            
    Returns:
        A `loadings(ctx)` function that returns the (n_var, n_latent)
        loading matrix
    """
    def loadings(ctx: _Context):
        shape = (ctx.n_var, ctx.n_latent)
        
        c_sq = numpyro.sample("c_sq", dist.InverseGamma(slab_df / 2, slab_df * slab_scale**2 / 2))
        
        tau = _half_cauchy_reparam("tau", tau0) # global shrinkage
        lam = _half_cauchy_reparam("lambda", 1.0, shape=shape, event_dim=2) # local shrinkage
        tau_sq, lam_sq = tau**2, lam**2
        lam_regularized = (c_sq * lam_sq) / (c_sq + tau_sq * lam_sq) # regularization
        
        beta = numpyro.sample("beta", dist.Normal(0, 1).expand(shape).to_event(2))
        discrimination = beta * jnp.sqrt(lam_regularized * tau)
                
        return discrimination
    return loadings