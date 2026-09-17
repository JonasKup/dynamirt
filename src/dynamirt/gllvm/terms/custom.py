from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.handlers import scope

from ..context import ModelContext


@dataclass(frozen=True)
class CustomTerm:
    """An additive contribution defined by user-provided NumPyro/JAX code.

    Args:
        name: Unique term name; local NumPyro sites are prefixed with
            ``name + "."``. Choose distinct names across latent and DIF terms,
            including built-ins such as item_intercept and residuals.
        fn: Called as ``fn(ctx, n_targets)``; must return an array of shape
            ``(ctx.n_obs, n_targets)``. n_targets is n_latent for latent terms
            and n_var for full-rank/DIF terms. Use explicit broadcasting when
            sharing a contribution across targets.

    Use NumPyro primitives directly inside fn. Prediction for new data is the
    function's responsibility; parameter sites must retain compatible shapes
    when reusing posterior draws. Configuration can be captured in a closure
    or functools.partial.
    """

    name: str
    fn: Callable[[ModelContext, int], ArrayLike]

    def __call__(self, ctx: ModelContext, n_targets: int):
        with scope(prefix=self.name, divider="."):
            contribution = jnp.asarray(self.fn(ctx, n_targets))
        expected = (ctx.n_obs, n_targets)
        if contribution.shape != expected:
            raise ValueError(
                f"CustomTerm {self.name!r} returned {contribution.shape}; "
                f"expected {expected}"
            )
        return contribution
