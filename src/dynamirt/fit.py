from typing import Callable, Mapping

import jax
from jax.typing import ArrayLike

from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoGuide, AutoNormal
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.mcmc import MCMCKernel
from numpyro.optim import Adam
from numpyro.handlers import block

import arviz as az
from xarray import DataTree

def fit_mcmc(
    model: Callable,
    responses: ArrayLike,
    covariates: Mapping[str, ArrayLike] = None,
    kernel_class: type[MCMCKernel]=NUTS,
    kernel_kwargs: dict | None = None,
    mcmc_kwargs: dict | None = None,
    rng_key: ArrayLike | None = None,
    return_deterministic: bool = True
    ) -> tuple[DataTree, MCMC]:

    """Run MCMC inference on a NumPyro model and return an xarray.DataTree.

    Args:
        model: A NumPyro model function with signature
            ``model(responses, covariates, ...)``.
        responses: Observation matrix passed to the model.
        covariates: Covariate mapping passed to the model. Default None.
        kernel_class: MCMC kernel class. Defaults to ``NUTS``.
        kernel_kwargs: Extra keyword arguments forwarded to the kernel
            constructor. Defaults to an empty dict.
        mcmc_kwargs: Keyword arguments forwarded to ``MCMC``. Defaults
            to ``{"num_warmup": 1500, "num_samples": 500, "num_chains": 4}``.
        rng_key: JAX PRNG key. Defaults to ``PRNGKey(0)``.
        return_deterministic: Should deterministic sites be included in
            the trace. Default True.

    Returns:
        A tuple (idata, mcmc) where idata is an
        ``arviz.InferenceData`` object and mcmc is the fitted
        ``numpyro.infer.MCMC`` instance.
    """
    
    covariates = {} if covariates is None else covariates
    
    kernel_kwargs = kernel_kwargs or {}
    mcmc_kwargs = mcmc_kwargs or {"num_warmup": 1500, "num_samples": 500, "num_chains": 4}
    rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)

    # record deterministic sites? Useful for decreasing RAM usage.
    _hide_deterministic = lambda site: site["type"] == "deterministic"
    fit_model = model if return_deterministic else block(model, hide_fn=_hide_deterministic)

    mcmc = MCMC(kernel_class(fit_model, **kernel_kwargs), **mcmc_kwargs)
    mcmc.run(rng_key, responses, covariates)

    idata = az.from_numpyro(mcmc)

    return idata, mcmc


def fit_svi(
    model: Callable,
    responses: ArrayLike,
    covariates: Mapping[str, ArrayLike] = None,
    guide_class: type[AutoGuide] = AutoNormal,
    guide_kwargs: dict | None = None,
    optim_kwargs: dict | None = None,
    run_kwargs: dict | None = None,
    rng_key: ArrayLike | None = None,
    num_samples: int=500,
    return_deterministic: bool=True
    ) -> tuple[DataTree, AutoGuide, SVIRunResult]:

    """Run stochastic variational inference (SVI) on a NumPyro model.

    After optimisation, draws posterior samples from the fitted guide and
    packages them into an xarray.DataTree.

    Args:
        model: A NumPyro model function with signature
            ``model(responses, covariates, ...)``.
        responses: Observation matrix passed to the model.
        covariates: Covariate mapping passed to the model. Default None.
        guide_class: Autoguide class used to construct the variational
            family. Defaults to ``AutoNormal``.
        guide_kwargs: Extra keyword arguments forwarded to the guide
            constructor. Defaults to an empty dict.
        optim_kwargs: Keyword arguments forwarded to the ``Adam``
            optimiser. Defaults to ``{"step_size": 1e-4}``.
        run_kwargs: Keyword arguments controlling the SVI run. Must
            contain ``"num_steps"``. Defaults to ``{"num_steps": 5000}``.
        rng_key: JAX PRNG key. Defaults to ``jax.random.key(0)``.
        num_samples: Number of posterior samples drawn from the fitted
            guide for the returned InferenceData. Defaults to 500.
        return_deterministic: Should deterministic sites be included in
            the trace. Default True.

    Returns:
        A tuple (idata, guide, svi_result) where idata is an
        ``arviz.InferenceData`` object, guide is the fitted autoguide
        instance, and svi_result is the ``SVIRunResult`` returned by
        ``svi.run``.
    """
    covariates = {} if covariates is None else covariates

    guide_kwargs = guide_kwargs or {}
    optim_kwargs = optim_kwargs or {"step_size": 1e-3}
    run_kwargs = run_kwargs or {"num_steps": 5000}
    rng_key = rng_key if rng_key is not None else jax.random.key(0)

    # record deterministic sites? Useful for decreasing RAM usage.
    _hide_deterministic = lambda site: site["type"] == "deterministic"
    fit_model = model if return_deterministic else block(model, hide_fn=_hide_deterministic)
    
    guide = guide_class(fit_model, **guide_kwargs)
    optim = Adam(**optim_kwargs)
    svi = SVI(fit_model, guide, optim, Trace_ELBO())
    svi_result = svi.run(rng_key, run_kwargs["num_steps"], responses, covariates)

    idata = az.from_numpyro_svi(
        svi,
        svi_result=svi_result,
        num_samples=num_samples,
        model_kwargs={"responses": responses, "covariates": covariates}
        )
    
    # add predictive
    
    return idata, guide, svi_result
