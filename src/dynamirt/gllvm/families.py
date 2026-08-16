from typing import Callable

from jax.typing import ArrayLike
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist 

from ._context import _Context


def Bernoulli() -> Callable:
    """Family factory for binary responses using a logit link.

    Returns:
        A family function with signature
            ``family(eta, ctx) -> dist.Bernoulli`` where eta is the
            linear predictor on the logit scale.
    """
    def family(eta: ArrayLike, ctx: _Context):
        return dist.Bernoulli(logits=eta)
    return family

def Gaussian(sigma_prior: dist.Distribution | None=None) -> Callable:
    """Family factory for continuous responses with a Gaussian likelihood.

    A per-variable standard deviation sigma is sampled from `sigma_prior`.

    Args:
        sigma_prior: Prior on the observation-level standard deviation.
            Expanded to shape (1, n_var). Defaults to HalfNormal(1) if None.

    Returns:
        A family function with signature
            ``family(eta, ctx) -> dist.Normal`` where eta is the
            conditional mean.
    """
    sigma_prior = dist.HalfNormal(1) if sigma_prior is None else sigma_prior
    
    def family(eta: ArrayLike, ctx: _Context):
        sigma = numpyro.sample("sigma", sigma_prior.expand((1, ctx.n_var)).to_event(2))
        return dist.Normal(eta, sigma)
    return family

def NegBinom(conc_prior: dist.Distribution | None=None) -> Callable:
    """Family factory for overdispersed count responses using a log link.

    A per-variable concentration (inverse overdispersion) parameter is
    sampled from `conc_prior`.

    Args:
        conc_prior: Prior on the concentration parameter. Expanded to
            shape (1, n_var). Defaults to HalfNormal(1) if None.

    Returns:
        A family function with signature
            ``family(eta, ctx) -> dist.NegativeBinomial2`` where eta is
            the linear predictor on the log scale.
    """
    
    conc_prior = dist.HalfNormal(1) if conc_prior is None else conc_prior

    def family(eta: ArrayLike, ctx: _Context):
        conc = numpyro.sample("concentration", conc_prior.expand((1, ctx.n_var)).to_event(2))
        return dist.NegativeBinomial2(jnp.exp(eta), conc)
    return family