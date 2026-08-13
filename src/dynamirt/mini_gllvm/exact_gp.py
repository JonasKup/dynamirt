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

@dataclass(frozen=True)
class Param:
    prior: dist.Distribution | float
    by_group: bool = False
    by_variable: bool = False
    per_predictor: bool = False  # ARD: one value per input dimension

    def __call__(self, name, n_groups, n_target, dim):
        tail = (dim,) if self.per_predictor else ()
        full = (n_groups, n_target) + tail
        if not isinstance(self.prior, dist.Distribution):
            return jnp.broadcast_to(jnp.asarray(self.prior), full)
        shape = (n_groups if self.by_group else 1, n_target if self.by_variable else 1) + tail
        v = numpyro.sample(name, self.prior.expand(shape).to_event(len(shape)))
        return jnp.broadcast_to(v, full)

def _pad_by_group(X, idx, n_groups):
    """create padded (T_max, n_groups, dim) array from (n_obs, dim)"""
    order = np.argsort(idx, kind='stable')
    sorted_idx = idx[order]

    n_time = np.bincount(idx, minlength=n_groups).max()

    group_pos = np.arange(idx.size) - np.searchsorted(sorted_idx, sorted_idx, side='left')

    padded = np.zeros((n_time, n_groups, X.shape[1]), dtype=X.dtype)
    valid = np.zeros((n_time, n_groups), bool)
    slot = np.empty_like(idx)

    padded[group_pos, sorted_idx] = X[order]
    valid[group_pos, sorted_idx] = True
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
        padded, valid, slot = _pad_by_group(X, group_idx, n_groups)
        T, dim = padded.shape[0], padded.shape[-1]

        n_kernels = n_target if any(q.by_variable for q in self.params.values()) else 1
        params = {k: q(f"{self.name}_{k}", n_groups, n_kernels, dim)
                for k, q in self.params.items()}
        z = numpyro.sample(f"{self.name}_z", dist.Normal(0, 1)
                        .expand((T, n_groups, n_target)).to_event(3))

        eye = jnp.eye(T)

        def cholesky(p, Xg, vg):
            K = self.kernel(p)(Xg, Xg)
            K = jnp.where(vg[:, None] & vg[None, :], K, eye)
            return jnp.linalg.cholesky(K + self.jitter * eye)

        L = jax.vmap(jax.vmap(cholesky, (0, None, None)))(
            params,
            jnp.asarray(np.moveaxis(padded, 0, 1)),   # (n_groups, T, dim)
            jnp.asarray(valid.T),                     # (n_groups, T)
        )                                             # (n_groups, n_kernels, T, T)

        spec = "gts,sgv->tgv" if n_kernels == 1 else "gvts,sgv->tgv"
        f = jnp.einsum(spec, L[:, 0] if n_kernels == 1 else L, z)
        f = numpyro.deterministic(self.name, f)       # (T, n_groups, n_target)
        return f[slot, group_idx]                     # (n_obs, n_target)
        
    def matern(nu=2.5, amplitude=1.0, length=dist.InverseGamma(5, 5), **flags):
        k = {0.5: kernels.Matern32, 1.5: kernels.Matern32, 2.5: kernels.Matern52}[nu]
        return dict(kernel=lambda p: p["amplitude"] ** 2 * k(p["length"]),
                    params={"amplitude": Param(amplitude, **flags),
                            "length": Param(length, **flags)})