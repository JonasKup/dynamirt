import numpy as np

import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp

from jax.typing import ArrayLike

from ._context import _Context

def Confirmatory(Q: ArrayLike):
    def loadings(ctx: _Context):
        rows, cols = np.nonzero(np.asarray(Q))

        loadings = numpyro.sample(
            "anchor_loadings",
            dist.Normal(0, 1).expand((rows.size,)).to_event(1),
        )
        
        discrimination = jnp.zeros((ctx.n_var, ctx.n_latent), dtype=loadings.dtype).at[rows, cols].set(loadings)

        return discrimination
    
    return loadings


def Unconstrained():
    def loadings(ctx: _Context):
        return numpyro.sample(
            "lam_raw", dist.Normal(0, 1).expand((ctx.n_var, ctx.n_latent)).to_event(2)
        )
        
    return loadings

# minimal sparse differentiable exploratory set up using horseshoe priors
def Exploratory():
    def exploratory(ctx: _Context):
        
        # global shrinkage
        tau = numpyro.sample("tau", dist.HalfCauchy(1))
        
        # local shrinkage
        lam = numpyro.sample("lambda", dist.HalfCauchy(1).expand((ctx.n_var, ctx.n_latent)).to_event(2))
        beta = numpyro.sample("beta", dist.Normal(0, 1).expand((ctx.n_var, ctx.n_latent)).to_event(2))
        
        discrimination = beta * lam * tau
                
        return discrimination
    return exploratory