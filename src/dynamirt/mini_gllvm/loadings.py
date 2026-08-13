import numpy as np

import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp

from jax.typing import ArrayLike

from ._context import _Context

def Fixed():
    # disables differential variable discrimination
    def loadings(ctx: _Context):
        
        return jnp.ones((ctx.n_var, ctx.n_latent))
    
    return loadings

def Confirmatory(
    Q: ArrayLike, 
    positive_anchors: ArrayLike = None, 
    free_prior: dist.Distribution | None = None,
    positive_prior: dist.Distribution | None = None
    ):
    
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
            l = numpyro.sample("loadings_confirmatory_free", free_prior.expand((n_free,)).to_event(1))
            discrimination = discrimination.at[rows[free], cols[free]].set(l)

        if n_pos:
            lp = numpyro.sample("loadings_confirmatory_positive", positive_prior.expand((n_pos,)).to_event(1))
            discrimination = discrimination.at[rows[pos], cols[pos]].set(lp)

        return discrimination

    return loadings

def Unconstrained(prior: dist.Distribution | None = None):
    prior = dist.Normal(0, 1) if prior is None else prior
    def loadings(ctx: _Context):
        return numpyro.sample(
            "loadings_unconstrained", prior.expand((ctx.n_var, ctx.n_latent)).to_event(2)
        )
        
    return loadings

def _half_cauchy_reparam(name, scale, shape=(), event_dim=0):
    """inv gamma reparametrization for half cauchy under SVI"""
    aux = numpyro.sample(f"{name}_aux", dist.InverseGamma(0.5, 1.0 / scale**2).expand(shape).to_event(event_dim))
    x_sq = numpyro.sample(name, dist.InverseGamma(0.5, 1.0 / aux).to_event(event_dim))
    return jnp.sqrt(x_sq)

def Sparsity(tau0=1.0, slab_scale: float=1.0, slab_df: float=4.0):
    """regularized horseshoe sparsity prior on loadings matrix.

    Juho Piironen. Aki Vehtari (2017). Electron. J. Statist. 11 (2) 5018 - 5051
    https://doi.org/10.1214/17-EJS1337SI 
    
    Args:
        tau0: Global shrinkage scale. p0 / (D - p0) * sigma / sqrt(n). see eq. 3.12.
        slab_scale: Typical magnitude of a loading that escapes shrinkage.
        slab_df: Degrees of freedom of the Student-t slab.
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