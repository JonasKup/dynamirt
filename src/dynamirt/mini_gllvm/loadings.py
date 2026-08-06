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

def Confirmatory(Q: ArrayLike, positive_anchors: ArrayLike = None):
    rows, cols = np.nonzero(np.asarray(Q))
    pos = (np.zeros(rows.size, bool) if positive_anchors is None
           else np.asarray(positive_anchors)[rows, cols].astype(bool))
    free = ~pos
    n_free, n_pos = int(free.sum()), int(pos.sum())

    def loadings(ctx: _Context):
        discrimination = jnp.zeros((ctx.n_var, ctx.n_latent))

        if n_free:
            l = numpyro.sample("loadings_confirmatory_free", dist.Normal(0, 1).expand((n_free,)).to_event(1))
            discrimination = discrimination.at[rows[free], cols[free]].set(l)

        if n_pos:
            lp = numpyro.sample("loadings_confirmatory_positive", dist.HalfNormal(1).expand((n_pos,)).to_event(1))
            discrimination = discrimination.at[rows[pos], cols[pos]].set(lp)

        return discrimination

    return loadings


def Unconstrained():
    def loadings(ctx: _Context):
        return numpyro.sample(
            "loadings_unconstrained", dist.Normal(0, 1).expand((ctx.n_var, ctx.n_latent)).to_event(2)
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