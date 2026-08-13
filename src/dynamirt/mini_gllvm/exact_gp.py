from dataclasses import dataclass
from typing import Sequence, Callable, Mapping

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import jax

from tinygp import kernels

import numpy as np

from ._context import _Context
from .terms import _factorize, _design

# @dataclass(frozen=True)
# class Pool:
#     loc: dist.Distribution
#     scale: dist.Distribution

@dataclass(frozen=True)
class Param:
    """
    Shape helper. Terms (e.g. GPs) may extend along groups and/or variables/latents.
    These terms may contain parameters (this class) that might themselves be shared across axes.
    This class takes either scalars or distributions and broadcasts them to the specified shapes.
    """
    prior: dist.Distribution | float
    by_group: bool = False
    by_variable: bool = False

    def __call__(self, name, n_groups, n_target):
        
        full = (n_groups, n_target)
        
        # parameter is a scalar and is repeated across groups and variables/latents 
        if not isinstance(self.prior, dist.Distribution):
            deterministic = jnp.broadcast_to(jnp.asarray(self.prior), full)
            return numpyro.deterministic(name, deterministic)
        
        # determine shape of parameter and expand + broadcast accordingly
        shape = (n_groups if self.by_group else 1, n_target if self.by_variable else 1)
        random = numpyro.sample(name, self.prior.expand(shape).to_event(len(shape)))
        
        return jnp.broadcast_to(random, full)


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

    def __call__(self, ctx, n_vars):
        group_idx, n_groups = _factorize(ctx, self.group_by)
        n_target = n_vars if self.varies_over_variables else 1

        X = _design(ctx, self.predictors)
        # pad ragged group points into common shape of n_max_points for efficient batching
        padded, valid, slot = _pad_by_group(X, group_idx, n_groups)
        n_points = padded.shape[0]

        # special handling for case in which no parameters vary over any variables -> no need to vmap over variables
        n_kernels = n_target if any(q.by_variable for q in self.params.values()) else 1
        
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
        
# def matern(nu=2.5, amplitude=1.0, length=dist.InverseGamma(5, 5), **flags):
#     k = {0.5: kernels.Matern32, 1.5: kernels.Matern32, 2.5: kernels.Matern52}[nu]
#     return dict(kernel=lambda p: p["amplitude"] ** 2 * k(p["length"]),
#                 params={"amplitude": Param(amplitude, **flags),
#                         "length": Param(length, **flags)})