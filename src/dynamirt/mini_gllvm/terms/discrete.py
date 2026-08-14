from dataclasses import dataclass

from .._context import _Context
from ..parameters import Param
#from ..terms import _factorize, _design

import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp
import jax

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
        group_idx, n_groups = ctx._factorize(self.group_by)
        t_idx, n_time = ctx._factorize(self.order_by)
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
