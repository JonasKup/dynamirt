from dataclasses import dataclass
from typing import Sequence, Callable, Mapping, Literal

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern, diag_spectral_density_squared_exponential
import jax.numpy as jnp
import jax

from tinygp import kernels

import numpy as np

from .._context import _Context
from ..parameters import Param

#------------------------------------------ Exact GPs

def _pad_by_group(X, idx, n_groups):
    """create padded (n_max_points, n_groups, dim) array from (n_obs, dim)"""
    order = np.argsort(idx, kind='stable')
    sorted_idx = idx[order]

    n_max_points = np.bincount(idx, minlength=n_groups).max()

    group_pos = np.arange(idx.size) - np.searchsorted(sorted_idx, sorted_idx, side='left')

    padded = np.zeros((n_groups, n_max_points, X.shape[1]), dtype=X.dtype)
    valid = np.zeros((n_groups, n_max_points), bool)
    slot = np.empty_like(idx)

    padded[sorted_idx, group_pos] = X[order]
    valid[sorted_idx, group_pos] = True
    slot[order] = group_pos
    
    return padded, valid, slot

@dataclass(frozen=True)
class ExactGP:
    name: str
    predictors: str | Sequence[str]
    kernel: Callable[[dict], kernels.Kernel]
    params: Mapping[str, Param]
    group_by: str | None = None
    varies_over_variables: bool = True
    jitter: float = 1e-6

    def __call__(self, ctx: _Context, n_vars: int):
        group_idx, n_groups = ctx._factorize(self.group_by)
        n_target = n_vars if self.varies_over_variables else 1

        X = ctx._design(self.predictors)
        # pad ragged group points into common shape of n_max_points for efficient batching
        padded, valid, slot = _pad_by_group(X, group_idx, n_groups)
        n_points = padded.shape[1]

        # special handling for case in which no parameters vary over any variables -> no need to vmap over variables
        #n_kernels = n_target if any(q.by_variable for q in self.params.values()) else 1
        n_kernels = n_target if any(q.by_variable != "shared" for q in self.params.values()) else 1
        
        kernel_params = {k: q(f"{self.name}_{k}", n_groups, n_kernels) for k, q in self.params.items()} # (n_groups, n_kernels)

        # GP draws. Multiplied with kernel Cholesky
        z = numpyro.sample(f"{self.name}_z", dist.Normal(0, 1).expand((n_points, n_groups, n_target)).to_event(3))

        identity = jnp.eye(n_points)

        def cholesky(kernel_params, padded_input, valid):
            covariance = self.kernel(kernel_params)(padded_input, padded_input)
            # set padded regions to identity
            covariance = jnp.where(valid[:, None] & valid[None, :], covariance, identity)
            return jnp.linalg.cholesky(covariance + self.jitter * identity)

        # vmap over groups (outer) and kernels/variables (inner)
        L = jax.vmap(jax.vmap(cholesky, (0, None, None)))(kernel_params, padded, valid) # (n_groups, n_kernels, n_points, n_points)

        spec = "gts,sgv->tgv" if n_kernels == 1 else "gvts,sgv->tgv"
        f = jnp.einsum(spec, L[:, 0] if n_kernels == 1 else L, z)
        
        f = numpyro.deterministic(self.name, f)       # (n_points, n_groups, n_target)
        return f[slot, group_idx]                     # (n_obs, n_target)
        
#------------------------------ HSGPs

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

    amplitude: Param = Param(1.0)
    length: Param = Param(dist.InverseGamma(5, 5), by_variable="free")

    def __call__(self, ctx: _Context, n_vars: int):
        idx, n_groups = ctx._factorize(self.group_by)
        n_target = n_vars if self.varies_over_variables else 1
        X = ctx._design(self.predictors)

        alpha = self.amplitude(f"{self.name}_amplitude", n_groups, n_target)
        length = self.length(f"{self.name}_length", n_groups, n_target)

        phi, weights = _hsgp_kernel(X, self.nu, alpha, length, self.ell, self.m,
                                    self.name, self.kernel)
        f = jnp.einsum("nb,nvb->nv", phi, weights[idx])
        return numpyro.deterministic(self.name, f)