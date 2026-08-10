from numpyro.infer import MCMC, NUTS

from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
import jax

import arviz as az

def fit_mcmc(
    model,
    responses,
    covariates,
    kernel_class=NUTS,
    kernel_kwargs=None,
    mcmc_kwargs=None,
    rng_key=None,
    ):

    kernel_kwargs = kernel_kwargs or {}
    mcmc_kwargs = mcmc_kwargs or {"num_warmup": 1500, "num_samples": 500}
    rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)

    mcmc = MCMC(kernel_class(model, **kernel_kwargs), **mcmc_kwargs)
    mcmc.run(rng_key, responses, covariates)

    idata = az.from_numpyro(mcmc)

    return idata, mcmc


def fit_svi(
    model,
    responses,
    covariates,
    guide_class=AutoNormal,
    guide_kwargs=None,
    optim_kwargs=None,
    run_kwargs=None,
    rng_key=None,
    num_samples: int=500
    ):

    guide_kwargs = guide_kwargs or {}
    optim_kwargs = optim_kwargs or {"step_size": 1e-4}
    run_kwargs = run_kwargs or {"num_steps": 5000}
    rng_key = rng_key if rng_key is not None else jax.random.key(0)

    guide = guide_class(model, **guide_kwargs)
    optim = Adam(**optim_kwargs)
    svi = SVI(model, guide, optim, Trace_ELBO())
    svi_result = svi.run(rng_key, run_kwargs["num_steps"], responses, covariates)

    idata = az.from_numpyro_svi(
        svi,
        svi_result=svi_result,
        num_samples=num_samples,
        model_kwargs={"responses": responses, "covariates": covariates}
        )
    
    # add predictive
    
    return idata, guide, svi_result
