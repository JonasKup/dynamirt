from ._context import _Context

from jax.typing import ArrayLike
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist 

def Bernoulli():
    def family(eta: ArrayLike, ctx: _Context):
        return dist.Bernoulli(logits=eta)
    return family

def Gaussian(sigma_prior=dist.HalfNormal(1)):
    def family(eta: ArrayLike, ctx: _Context):
        sigma = numpyro.sample("sigma", sigma_prior.expand((1, ctx.n_var)))
        return dist.Normal(eta, sigma)
    return family

def NegBinom(conc_prior=dist.HalfNormal(1)):
    def family(eta: ArrayLike, ctx: _Context):
        conc = numpyro.sample("concentration", conc_prior.expand((1, ctx.n_var)))
        return dist.NegativeBinomial2(jnp.exp(eta), conc)
    return family