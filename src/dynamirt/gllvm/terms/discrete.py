from dataclasses import dataclass, field

from .._context import _Context
from ..parameters import Param

import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp
import jax

@dataclass(frozen=True)
class GRW:
    """Gaussian random walk over `order_by`, independent per `group_by` level.

    sigma is the innovation standard deviation: the step size of the walk.
    Assumes equally spaced time steps with none missing. Emits (n_obs, n_target).
    """
    name: str
    order_by: str
    group_by: str | None = None
    varies_over_variables: bool = True
    scale: Param = Param(dist.HalfNormal(1))

    def __call__(self, ctx: _Context, n_vars: int):
        group_idx, n_groups = ctx._factorize(self.group_by)
        t_idx, n_time = ctx._factorize(self.order_by)
        n_target = n_vars if self.varies_over_variables else 1

        sigma = self.scale(f"{self.name}_scale", n_groups, n_target)  # (n_groups, n_target)
        z = numpyro.sample(f"{self.name}_innovations", dist.Normal(0, 1).expand((n_time, n_groups, n_target)).to_event(3))

        x = numpyro.deterministic(self.name, jnp.cumsum(sigma * z, axis=0))
        return x[t_idx, group_idx]                                   # (n_obs, n_target)

@dataclass(frozen=True)
class AR1:
    """Stationary first-order autoregressive process over `order_by`.

    phi is the persistence (correlation between consecutive steps), sigma the
    innovation standard deviation. The x_1 scale is the stationary marginal
    standard deviation, so the process has constant variance sigma^2/(1 - phi^2)
    throughout rather than warming up from zero.
    Assumes equally spaced time steps with none missing. Emits (n_obs, n_target).
    """
    name: str
    order_by: str
    group_by: str | None = None
    varies_over_variables: bool = True
    scale: Param = Param(dist.HalfNormal(1))
    phi: Param = Param(dist.Beta(3, 3))

    def __call__(self, ctx: _Context, n_vars: int):
        group_idx, n_groups = ctx._factorize(self.group_by)
        t_idx, n_time = ctx._factorize(self.order_by)
        n_target = n_vars if self.varies_over_variables else 1

        phi = self.phi(f"{self.name}_phi", n_groups, n_target)       # (n_groups, n_target)
        sigma = self.scale(f"{self.name}_scale", n_groups, n_target)
        eps = sigma * numpyro.sample(f"{self.name}_innovations",dist.Normal(0, 1).expand((n_time, n_groups, n_target)).to_event(3))
        x0 = eps[0] / jnp.sqrt(1 - phi ** 2)

        def step(carry, e):
            carry = phi * carry + e
            return carry, carry

        _, xs = jax.lax.scan(step, x0, eps[1:])
        x = numpyro.deterministic(self.name, jnp.concatenate([x0[None], xs]))
        return x[t_idx, group_idx]