from .mini_gllvm._context import _Context

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import numpyro.distributions as dist
import numpyro

from typing import Literal

def _irt_NPL(
    model_type: Literal["3PL", "4PL"],
    lower_asymptote_prior: dist.Distribution = None,
    upper_asymptote_prior: dist.Distribution = None
    ):
    
    """Compute likelihood for 3PL and 4PL models"""
    
    lower_asymptote_prior = dist.Beta(2, 8) if lower_asymptote_prior is None else lower_asymptote_prior
    upper_asymptote_prior = dist.Beta(8, 2) if upper_asymptote_prior is None else upper_asymptote_prior

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

