from dataclasses import dataclass, field

from .._context import _Context
from ..parameters import Param

import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp
import jax

@dataclass(frozen=True)
class GRW:
    """Gaussian random walk term for a gllvm regression.

    At each time step the process increments by a Normal(0, sigma) draw.
    Assumes equally spaced, fully observed time steps within each group.

    Attributes:
        name: Sample-site prefix. The cumulative walk is stored under a
            deterministic site with this name.
        order_by: Covariate name whose unique sorted values define the
            time axis.
        group_by: Optional covariate name giving a grouping factor. An
            independent walk is drawn per level. Defaults to None.
        varies_over_variables: If True, an independent walk is drawn for
            each response variable. Defaults to True.
        scale: ``Param`` for the innovation standard deviation sigma.
            Defaults to HalfNormal(1).
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
    """Stationary first-order autoregressive term for a gllvm regression.

    The process is initialised at its stationary marginal standard
    deviation ``sigma / sqrt(1 - phi^2)`` so that variance is constant
    across time rather than warming up from zero.
    Assumes equally spaced, fully observed time steps within each group.

    Attributes:
        name: Sample-site prefix. The AR(1) trajectory is stored under
            a deterministic site with this name.
        order_by: Covariate name whose unique sorted values define the
            time axis.
        group_by: Optional covariate name giving a grouping factor. An
            independent process is drawn per level. Defaults to None.
        varies_over_variables: If True, an independent process is drawn
            for each response variable. Defaults to True.
        scale: ``Param`` for the innovation standard deviation sigma.
            Defaults to HalfNormal(1).
        phi: ``Param`` for the autoregressive persistence coefficient
            (correlation between consecutive steps). Defaults to
            Beta(3, 3), which is symmetric on (0, 1).
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