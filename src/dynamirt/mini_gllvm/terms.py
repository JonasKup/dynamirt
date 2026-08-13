from dataclasses import dataclass
from typing import Literal, Sequence

from ._context import _Context

import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern, diag_spectral_density_squared_exponential

import jax.numpy as jnp
import jax

def _factorize(ctx, key):
    """interpret covariate as index column for grouping"""
    if key is None:
        return np.zeros(ctx.n_obs, int), 1
    codes, idx = np.unique(np.asarray(ctx.covariates[key]).ravel(), return_inverse=True)
    return idx, codes.size

def _design(ctx, predictors):
    """stack multiple covariates into single array if multiple were given. 
    Broadcast scalar values to correct length."""
    keys = [predictors] if isinstance(predictors, str) else list(predictors)
    cols = [np.broadcast_to(np.asarray(ctx.covariates[k]).ravel(), (ctx.n_obs,)) for k in keys]
    return np.stack(cols, axis=-1)

@dataclass(frozen=True)
class Linear:
    
    """Linear term `X @ coef` for a gllvm regression.

    Contributes (n_obs, n_target) where n_target is number of variables (full rank), number of latents (reduced rank),
    or 1 broadcast over the stack if `varies_over_variables` is False.

    Attributes:
        name: sample-site prefix; the assembled coefficient is stored as a
            deterministic site under this name.
        predictors: covariate key(s) to regress on
        group_by: covariate key giving a grouping factor; a separate coefficient
            is drawn per level. None means a single level.
        constraint: "reference_coding" fixes the first group level to zero.
        pool_over_groups: share hyperparameters across group levels.
        pool_over_variables: share hyperparameters across variables/latents.
        varies_over_variables: draw one coefficient per variable/latent rather
            than a single one broadcast over all of them.
        corr: "predictors" correlates coefficients across covariates, "variables"
            across variables/latents, via an LKJ Cholesky factor. "both" correlates
            across covariates AND variables/latents.
        prior: prior on the raw coefficient (a non-centered offset when pooling).
        loc_prior: prior on the hyper-mean; used only when pooling.
        scale_prior: prior on the hyper-scale; used only when pooling.
    """
    
    name: str
    predictors: str | Sequence[str]
    group_by: str | None = None
    
    # todo: sum to zero
    constraint: Literal[None, "reference_coding"] = None
    
    #pool
    pool_over_groups: bool = False
    pool_over_variables: bool = False
    varies_over_variables: bool = True

    corr: Literal[None, "predictors", "variables", "both"] = None
    
    prior: dist.Distribution = dist.Normal(0,1) # prior on coefficient or coefficient-delta if pooling
    loc_prior: dist.Distribution = dist.Normal(0, 1)
    scale_prior: dist.Distribution = dist.HalfNormal(1)


    def __call__(self, ctx: _Context, n_vars: int):
        # n_vars is either n_var or n_latents depending on whether Linear is called for eta or u regression

        X = _design(ctx, self.predictors)
        n_predictors = X.shape[1]

        idx, n_groups = _factorize(ctx, self.group_by)
                   
        # parameter per 'stacked glm' or single parameter broadcast over glm stack
        # if not varies_over_variables this is another entry point for latent variables that are not subject to the loadings matrix
        n_target = n_vars if self.varies_over_variables else 1

        # sampling one fewer level for reference coding
        n_free = n_groups - 1 if self.constraint == "reference_coding" else n_groups
        shape = (n_free, n_target, n_predictors)
        
        # hyperparameters are shared (size 1) along any axis we pool over
        hyper_shape = (1 if self.pool_over_groups else n_free, 1 if self.pool_over_variables else n_target, n_predictors)
        coef = numpyro.sample(f"{self.name}_raw", self.prior.expand(hyper_shape).to_event(3))

        # b = group_by-level, v = variable/latent, o = over
        if self.corr == "predictors":
            L = numpyro.sample(f"{self.name}_L", dist.LKJCholesky(n_predictors, 1))
            coef = jnp.einsum("bvo,po->bvp", coef, L)
        elif self.corr == "variables":
            L = numpyro.sample(f"{self.name}_L", dist.LKJCholesky(n_target, 1))
            coef = jnp.einsum("bvo,wv->bwo", coef, L)
        elif self.corr == "both":
            m = n_target * n_predictors
            L = numpyro.sample(f"{self.name}_L", dist.LKJCholesky(m, 1))   # (m, m) lower-triangular
            coef = (coef.reshape(n_free, m) @ L.T).reshape(n_free, n_target, n_predictors)
            
        # pooling
        if self.pool_over_groups or self.pool_over_variables:
            loc = numpyro.sample(f"{self.name}_loc", self.loc_prior.expand(hyper_shape).to_event(3))
            scale = numpyro.sample(f"{self.name}_scale", self.scale_prior.expand(hyper_shape).to_event(3))
            coef = loc + scale * coef

        if self.constraint == "reference_coding":
            coef = jnp.pad(coef, ((1, 0), (0, 0), (0, 0)))

        
        coef = numpyro.deterministic(self.name, coef) # (n_groups, n_target, n_predictors)
        # n = n_obs, o = over, v = variable/latent
        return jnp.einsum("no,nvo->nv", X, coef[idx]) # (n_obs, n_var)

@dataclass(frozen=True)
class _GaussMarkov:
    """Shared plumbing for discrete-time Gauss-Markov terms.

    Assumes equally spaced time steps with none missing. Emits (n_obs, n_target).
    """
    name: str
    order_by: str
    group_by: str | None = None
    scale_by_group: bool = False
    scale_by_variable: bool = False
    varies_over_variables: bool = True
    scale_prior: dist.Distribution | float = dist.HalfNormal(1)

    def _axes(self, ctx, n_vars):
        group_idx, n_groups = _factorize(ctx, self.group_by)
        t_idx, n_time = _factorize(ctx, self.order_by)
        n_target = n_vars if self.varies_over_variables else 1
        return group_idx, n_groups, t_idx, n_time, n_target

    def _scale(self, n_groups, n_target):
        if not isinstance(self.scale_prior, dist.Distribution):
            return self.scale_prior
        shape = (n_groups if self.scale_by_group else 1,
                 n_target if self.scale_by_variable else 1)
        return numpyro.sample(f"{self.name}_scale",
                              self.scale_prior.expand(shape).to_event(2))

    def _innovations(self, n_time, n_groups, n_target):
        return numpyro.sample(f"{self.name}_innovations",
                              dist.Normal(0, 1)
                              .expand((n_time, n_groups, n_target)).to_event(3))

@dataclass(frozen=True)
class GRW(_GaussMarkov):
    """Gaussian random walk. x_1 = sigma * z_1, x_t = x_{t-1} + sigma * z_t."""

    def __call__(self, ctx: _Context, n_vars: int):
        group_idx, n_groups, t_idx, n_time, n_target = self._axes(ctx, n_vars)
        scale = self._scale(n_groups, n_target)
        z = self._innovations(n_time, n_groups, n_target)
        x = numpyro.deterministic(self.name, jnp.cumsum(scale * z, axis=0))
        return x[t_idx, group_idx]

@dataclass(frozen=True)
class AR1(_GaussMarkov):
    """Stationary AR(1). x_1 ~ N(0, sigma / sqrt(1 - phi^2)), x_t = phi x_{t-1} + sigma z_t."""
    phi_by_group: bool = False
    phi_by_variable: bool = False
    phi_prior: dist.Distribution = dist.Beta(3, 3)

    def __call__(self, ctx: _Context, n_vars: int):
        group_idx, n_groups, t_idx, n_time, n_target = self._axes(ctx, n_vars)
        phi_shape = (n_groups if self.phi_by_group else 1,
                     n_target if self.phi_by_variable else 1)
        phi = numpyro.sample(f"{self.name}_phi",
                             self.phi_prior.expand(phi_shape).to_event(2))
        scale = self._scale(n_groups, n_target)
        eps = scale * self._innovations(n_time, n_groups, n_target)

        x0 = eps[0] / jnp.sqrt(1 - phi ** 2)

        def step(carry, e):
            carry = phi * carry + e
            return carry, carry

        _, xs = jax.lax.scan(step, x0, eps[1:])
        x = numpyro.deterministic(self.name, jnp.concatenate([x0[None], xs]))
        return x[t_idx, group_idx]

# hsgp adapted from https://num.pyro.ai/en/stable/_modules/numpyro/contrib/hsgp/approximation.html
# numpyro's hsgp helpers are scalar in alpha/length so manual vmap is required
def _hsgp_kernel(
    X: jax.Array,
    nu: float,
    alpha: jax.Array,
    length: jax.Array,
    ell: float | Sequence[float],
    m: int | Sequence[int],
    name: str,
    kernel: Literal["Matern", "ExpSquared"]
):
    dim = X.shape[-1]
    n_groups, n_target = alpha.shape

    phi = eigenfunctions(x=X, ell=ell, m=m) # (n_obs, n_basis)
    n_basis = phi.shape[-1]

    def _spd(length_k, alpha_k):
        if kernel == "Matern":
            return jnp.sqrt(diag_spectral_density_matern(
                nu=nu, alpha=alpha_k, length=length_k, ell=ell, m=m, dim=dim))
        return jnp.sqrt(diag_spectral_density_squared_exponential(
                alpha=alpha_k, length=length_k, ell=ell, m=m, dim=dim))

    spd = jax.vmap(_spd)(length.reshape(-1), alpha.reshape(-1)) # (n_groups*n_target, n_basis)
    spd = spd.reshape(n_groups, n_target, n_basis)

    beta = numpyro.sample(f"{name}_beta",
                          dist.Normal(0, 1)
                          .expand((n_groups, n_target, n_basis)).to_event(3))

    return phi, spd * beta # (n_obs, n_basis), (n_groups, n_target, n_basis)


@dataclass(frozen=True)
class HSGP:
    
    name: str
    predictors: str | Sequence[str]
    kernel: Literal["Matern", "ExpSquared"]
    nu: float
    ell: float | Sequence[float]
    m: int | Sequence[int]
    
    group_by: str | None = None
    varies_over_variables: bool = True
    
    amplitude_by_group: bool = False
    amplitude_by_variable: bool = True
    scale_by_group: bool = False
    scale_by_variable: bool = True
    
    amplitude_prior: dist.Distribution | float = 1.0
    scale_prior: dist.Distribution | float = dist.InverseGamma(5, 5)

    def __call__(self, ctx: _Context, n_vars: int):
        
        idx, n_groups = _factorize(ctx, self.group_by)
        n_target = n_vars if self.varies_over_variables else 1

        X = _design(ctx, self.predictors)
        
        alpha_shape = (n_groups if self.amplitude_by_group else 1, n_target if self.amplitude_by_variable else 1)
        scale_shape = (n_groups if self.scale_by_group else 1, n_target if self.scale_by_variable else 1)
        
        if isinstance(self.amplitude_prior, dist.Distribution):
            alpha = numpyro.sample(f"{self.name}_amplitude", self.amplitude_prior.expand(alpha_shape).to_event(2))
        else:
            alpha = jnp.full(alpha_shape, self.amplitude_prior)
            
        if isinstance(self.scale_prior, dist.Distribution):
            scale = numpyro.sample(f"{self.name}_scale", self.scale_prior.expand(scale_shape).to_event(2))
        else:
            scale = jnp.full(scale_shape, self.scale_prior)
            
        alpha = jnp.broadcast_to(alpha, (n_groups, n_target))
        scale = jnp.broadcast_to(scale, (n_groups, n_target))
                            
        phi, weights = _hsgp_kernel(X, self.nu, alpha, scale, self.ell, self.m, self.name, self.kernel)
        f = jnp.einsum("nb,nvb->nv", phi, weights[idx])
        return numpyro.deterministic(self.name, f)