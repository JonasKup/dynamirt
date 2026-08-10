from .mini_gllvm._context import _Context
from .mini_gllvm.families import Bernoulli

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import OrderedTransform

from typing import Literal

def _dichotomous(
    model_type: Literal["1PL", "2PL", "3PL", "4PL"],
    lower_asymptote_prior: dist.Distribution = None,
    upper_asymptote_prior: dist.Distribution = None
    ):
    
    """Compute likelihood for 3PL and 4PL models"""
    
    lower_asymptote_prior = dist.Beta(2, 8) if lower_asymptote_prior is None else lower_asymptote_prior
    upper_asymptote_prior = dist.Beta(8, 2) if upper_asymptote_prior is None else upper_asymptote_prior

    if model_type in ["1PL", "2PL"]:
        return Bernoulli()

    def family(eta: ArrayLike, ctx: _Context):
        

        la = numpyro.sample("lower_asymptote", lower_asymptote_prior.expand((ctx.n_var,)).to_event(1))

        if model_type == "4PL":
            # sample gap between ua and la so la > ua can never happen
            gap = numpyro.sample("asymptote_gap", upper_asymptote_prior.expand((ctx.n_var,)).to_event(1))
            ua = numpyro.deterministic("upper_asymptote", la + (1.0 - la) * gap)
        else:
            ua = 1.0

        p = la + (ua - la) * jax.nn.sigmoid(eta)
        p = jnp.clip(p, 1e-7, 1.0 - 1e-7) # guard the log-likelihood
        
        return dist.BernoulliProbs(probs=p)

    return family

# This might be flipping default sign of the intercept.
# prior naming needs to be cleared up.
# potentially separate GRM and GPCM
def _polytomous(
    model_type, 
    n_cat, 
    prior=None, 
    loc_prior=None, 
    gap_prior=None):
    
    prior = dist.Normal(0, 1) if prior is None else prior
    loc_prior = dist.Normal(0, 3) if loc_prior is None else loc_prior
    gap_prior = dist.Normal(0, 0.5) if gap_prior is None else gap_prior

    def family(eta, ctx):
        
        if model_type == "GRM":
            base = dist.Normal(
                jnp.concatenate([jnp.full((ctx.n_var, 1), loc_prior.loc),
                                 jnp.full((ctx.n_var, n_cat - 2), gap_prior.loc)], -1),
                jnp.concatenate([jnp.full((ctx.n_var, 1), loc_prior.scale),
                                 jnp.full((ctx.n_var, n_cat - 2), gap_prior.scale)], -1),
            ).to_event(1)
            c = numpyro.sample("cutpoints",
                               dist.TransformedDistribution(base, OrderedTransform()).to_event(1))
            return dist.OrderedLogistic(eta, c)
        
        base = prior.expand((ctx.n_var, n_cat - 1)).to_event(1)
        d = numpyro.sample("steps", base.to_event(1))
        logits = jnp.cumsum(jnp.pad(eta[..., None] - d, ((0, 0), (0, 0), (1, 0))), axis=-1)
        return dist.CategoricalLogits(logits)

    return family