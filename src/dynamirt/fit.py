from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import arviz as az
import numpy as np
import xarray as xr

import jax
from jax.typing import ArrayLike

from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoGuide, AutoNormal
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.svi import SVIRunResult
from numpyro.optim import Adam
from numpyro.handlers import block


@dataclass
class SVIState:
    """Fitted SVI state; the guide remains available through ``svi.guide``."""

    svi: SVI = field(repr=False)
    result: SVIRunResult = field(repr=False)


@dataclass
class FitResult:
    """Result returned by ``fit_mcmc`` or ``fit_svi``, with ArviZ conversion."""

    inference: MCMC | SVIState = field(repr=False)
    model: Callable = field(repr=False)
    responses: Any = field(repr=False)
    covariates: dict[str, Any] = field(repr=False)
    dims: dict[str, list[str]] = field(default_factory=dict)
    coords: dict[str, Any] = field(default_factory=dict)

    def to_idata(
        self,
        *,
        num_samples: int | None = None,
        dims: dict[str, list[str]] | None = None,
        coords: dict | None = None,
    ) -> xr.DataTree:
        """Convert available posterior results to labeled ArviZ 1.x data.

        Args:
            num_samples: Number of posterior draws for SVI. Defaults to 500.
                Leave as None for MCMC to retain all existing draws. Default None.
            dims: Site-to-dimension overrides, merged with the model defaults.
                Defaults to None.
            coords: Coordinate labels, merged with the model defaults.
                Defaults to None.

        Returns:
            An xarray.DataTree with named dimensions and observation-level
            covariates attached as coordinates.

        Raises:
            ValueError: If num_samples is supplied for MCMC 
                or a covariate name conflicts with an existing
                variable or dimension.
        """
        conversion_dims = {
            name: list(axes) for name, axes in {**self.dims, **(dims or {})}.items()
        }
        conversion_coords = {**self.coords, **(coords or {})}
        kwargs = dict(coords=conversion_coords, dims=conversion_dims)
        if isinstance(self.inference, MCMC):
            if num_samples is not None:
                raise ValueError("num_samples is only supported for SVI conversion")
            idata = az.from_numpyro(self.inference, pred_dims=conversion_dims, **kwargs)
        elif isinstance(self.inference, SVIState):
            if num_samples is None:
                num_samples = 500

            idata = az.from_numpyro_svi(
                self.inference.svi,
                svi_result=self.inference.result,
                num_samples=num_samples,
                model_args=(self.responses, self.covariates),
                pred_dims=conversion_dims,
                **kwargs,
            )
        else:
            raise TypeError("inference must be an MCMC object or SVIState")

        observation_coords = {
            name: ("obs", np.asarray(values))
            for name, values in self.covariates.items()
            if np.shape(values) == (np.shape(self.responses)[0],)
        }
        for name, group in list(idata.children.items()):
            if "obs" in group.dims:
                collisions = set(observation_coords) & (set(group.variables) | set(group.dims))
                if collisions:
                    raise ValueError(
                        f"Covariate names conflict with analysis variables/dimensions: {sorted(collisions)}"
                    )
                idata[name] = group.to_dataset().assign_coords(observation_coords)
        return idata


def _make_result(inference, model, responses, covariates):
    """Prepare model-owned metadata once, without inferring meanings by size."""
    site_dims = {name: list(axes) for name, axes in getattr(model, "_dynamirt_dims", {}).items()}
    labels = {}
    if site_dims:
        n_obs, n_items = np.shape(responses)
        labels = {
            "obs": np.arange(n_obs), "item": np.arange(n_items),
            "latent": np.arange(model._dynamirt_n_latent),
        }
    return FitResult(inference, model, responses, dict(covariates), site_dims, labels)


def fit_mcmc(
    model: Callable,
    responses: ArrayLike,
    covariates: Mapping[str, ArrayLike] = None,
    kernel_class: type[MCMCKernel]=NUTS,
    kernel_kwargs: dict | None = None,
    mcmc_kwargs: dict | None = None,
    rng_key: ArrayLike | None = None,
    return_deterministic: bool = True,
    *,
    run_kwargs: dict | None = None,
    ) -> FitResult:

    """Run MCMC inference on a dynamirt NumPyro model. Returns a FitResult with native sampler access.

    Args:
        model: A NumPyro model function with signature
            ``model(responses, covariates, ...)``.
        responses: Observation matrix passed to the model.
        covariates: Covariate mapping passed to the model. Default None.
        kernel_class: MCMC kernel class. Defaults to ``NUTS``.
        kernel_kwargs: Extra keyword arguments forwarded to the kernel
            constructor. Defaults to an empty dict.
        mcmc_kwargs: Keyword arguments forwarded to ``MCMC``. Overrides defaults
            to ``{"num_warmup": 1500, "num_samples": 500, "num_chains": 4}``.
        rng_key: JAX PRNG key. Defaults to ``PRNGKey(0)``.
        return_deterministic: Should deterministic sites be included in
            the fitting trace. Default True.
        run_kwargs: Extra keyword arguments forwarded to ``MCMC.run``, such as
            ``extra_fields`` or ``init_params``. Defaults to an empty dict.

    Returns:
        A FitResult. Access the sampler through ``result.inference`` and
        call ``result.to_idata()`` for labeled ArviZ data.
    """
    
    covariates = {} if covariates is None else covariates
    
    kernel_kwargs = kernel_kwargs or {}
    mcmc_kwargs = {"num_warmup": 1500, "num_samples": 500, "num_chains": 4, **(mcmc_kwargs or {})}
    rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)

    # record deterministic sites? Useful for decreasing RAM usage.
    _hide_deterministic = lambda site: site["type"] == "deterministic"
    fit_model = model if return_deterministic else block(model, hide_fn=_hide_deterministic)

    mcmc = MCMC(kernel_class(fit_model, **kernel_kwargs), **mcmc_kwargs)
    mcmc.run(rng_key, responses, covariates, **(run_kwargs or {}))

    return _make_result(mcmc, model, responses, covariates)


def fit_svi(
    model: Callable,
    responses: ArrayLike,
    covariates: Mapping[str, ArrayLike] = None,
    guide_class: type[AutoGuide] = AutoNormal,
    guide_kwargs: dict | None = None,
    optim: Callable | None = None,
    run_kwargs: dict | None = None,
    rng_key: ArrayLike | None = None,
    return_deterministic: bool=True,
    *,
    num_steps: int = 5000,
    ) -> FitResult:

    """Run stochastic variational inference (SVI) on a daynamirt NumPyro model.

    Returns native SVI state. Posterior draws are generated lazily when
    ``result.to_idata()`` is called.

    Args:
        model: A NumPyro model function with signature
            ``model(responses, covariates, ...)``.
        responses: Observation matrix passed to the model.
        covariates: Covariate mapping passed to the model. Default None.
        guide_class: Autoguide class used to construct the variational
            family. Defaults to ``AutoNormal``.
        guide_kwargs: Extra keyword arguments forwarded to the guide
            constructor. Defaults to an empty dict.
        optim: NumPyro optimizer. Defaults to ``Adam(step_size=1e-3)``.
        num_steps: Number of optimization steps. Defaults to 5000.
        run_kwargs: Extra options forwarded to ``SVI.run``, such as
            progress_bar or init_state. Defaults to an empty dict.
        rng_key: JAX PRNG key. Defaults to ``jax.random.key(0)``.
        return_deterministic: Should deterministic sites be included in
            the fitting trace. Default True.

    Returns:
        A FitResult whose ``inference`` is an SVIState containing the SVI
        object and optimization result.
    """
    covariates = {} if covariates is None else covariates

    guide_kwargs = guide_kwargs or {}
    run_kwargs = run_kwargs or {}
    rng_key = rng_key if rng_key is not None else jax.random.key(0)
    
    if optim is None:
        optim = Adam(step_size=1e-3)

    # record deterministic sites? Useful for decreasing RAM usage.
    _hide_deterministic = lambda site: site["type"] == "deterministic"
    fit_model = model if return_deterministic else block(model, hide_fn=_hide_deterministic)
    
    guide = guide_class(fit_model, **guide_kwargs)
    svi = SVI(fit_model, guide, optim, Trace_ELBO())
    svi_result = svi.run(rng_key, num_steps, responses, covariates, **run_kwargs)

    return _make_result(
        SVIState(svi, svi_result), model, responses, covariates,
    )
