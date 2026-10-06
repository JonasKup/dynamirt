from dataclasses import dataclass, field

from ..context import ModelContext
from ..parameters import Param

import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp
import jax

@dataclass(frozen=True)
class GRW:
    """Gaussian random walk term for a gllvm regression.

    At each time step the process increments by a Normal(0, sigma) draw.
    Uses an equally spaced latent time grid; observation rows may omit steps.

    The complete grid is stored locally as ``state``; 
    standardized draws are ``innovations.raw``.
    
    Attributes:
        name: Term identifier. The model scopes local sites under
            ``latent.<name>`` or ``item.<name>``. 
        order_by: Integer-coded time covariate indexing equally spaced steps.
            Declare the full time-axis size in index_sizes.
        group_by: Optional integer-coded grouping covariate; its size must
            be declared in index_sizes. An independent walk is drawn per level. Defaults to None.
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

    def __call__(self, ctx: ModelContext, n_vars: int):
        group_idx, n_groups = ctx.index(self.group_by)
        t_idx, n_time = ctx.index(self.order_by)
        n_target = n_vars if self.varies_over_variables else 1

        sigma = self.scale("scale", n_groups, n_target)  # (n_groups, n_target)
        z = numpyro.sample("innovations.raw", dist.Normal(0, 1).expand((n_time, n_groups, n_target)).to_event(3))

        x = numpyro.deterministic("state", jnp.cumsum(sigma * z, axis=0))
        return x[t_idx, group_idx]                                   # (n_obs, n_target)

@dataclass(frozen=True)
class AR1:
    """Stationary first-order autoregressive term for a gllvm regression.

    The process is initialised at its stationary marginal standard
    deviation ``sigma / sqrt(1 - phi^2)`` so that variance is constant
    across time rather than warming up from zero.
    Uses an equally spaced latent time grid; observation rows may omit steps.

    The complete grid is stored locally as ``state``; 
    standardized draws are ``innovations.raw``.

    Attributes:
        name: Term identifier. The model scopes local sites under
            ``latent.<name>`` or ``item.<name>``. 
        order_by: Integer-coded time covariate indexing equally spaced steps.
            Declare the full time-axis size in index_sizes.
        group_by: Optional integer-coded grouping covariate; its size must
            be declared in index_sizes. An independent process is drawn per level. Defaults to None.
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

    def __call__(self, ctx: ModelContext, n_vars: int):
        group_idx, n_groups = ctx.index(self.group_by)
        t_idx, n_time = ctx.index(self.order_by)
        n_target = n_vars if self.varies_over_variables else 1

        phi = self.phi("phi", n_groups, n_target)       # (n_groups, n_target)
        sigma = self.scale("scale", n_groups, n_target)
        eps = sigma * numpyro.sample("innovations.raw",dist.Normal(0, 1).expand((n_time, n_groups, n_target)).to_event(3))
        x0 = eps[0] / jnp.sqrt(1 - phi ** 2)

        def step(carry, e):
            carry = phi * carry + e
            return carry, carry

        _, xs = jax.lax.scan(step, x0, eps[1:])
        x = numpyro.deterministic("state", jnp.concatenate([x0[None], xs]))
        return x[t_idx, group_idx]