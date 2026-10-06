from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax.typing import ArrayLike

from ..context import ModelContext


@dataclass(frozen=True)
class CustomTerm:
    """An additive contribution defined by user-provided NumPyro/JAX code.

    Args:
        name: Term identifier. The enclosing model scopes local NumPyro
            sites under ``latent.<name>`` or ``item.<name>`` and records the
            output as ``contribution``.
        fn: Called as ``fn(ctx, n_targets)``; must return an array of shape
            ``(ctx.n_obs, n_targets)``. n_targets is n_latent for latent terms
            and n_var for full-rank/DIF terms. Use explicit broadcasting when
            sharing a contribution across targets.
    """

    name: str
    fn: Callable[[ModelContext, int], ArrayLike]

    def __call__(self, ctx: ModelContext, n_targets: int):
        contribution = jnp.asarray(self.fn(ctx, n_targets))
        expected = (ctx.n_obs, n_targets)
        if contribution.shape != expected:
            raise ValueError(
                f"CustomTerm {self.name!r} returned {contribution.shape}; "
                f"expected {expected}"
            )
        return contribution
