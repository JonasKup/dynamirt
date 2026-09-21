from __future__ import annotations

from typing import Literal
from dataclasses import dataclass, field

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp


@dataclass(frozen=True)
class Param:
    """Broadcastable parameter with optional non-centered partial pooling.

    Wraps a scalar constant or a prior distribution and manages its
    expansion along the group and variable/latent axes of a term. When
    either axis is set to a ``Pool`` instance, a non-centered
    parameterisation is used: a shared hyper-mean and hyper-scale are
    sampled, and the raw draw is shifted and scaled accordingly.

    Attributes:
        prior: Either a fixed scalar value (no sampling) or a
            ``numpyro.distributions.Distribution`` used as the prior.
        by_group: ``"shared"`` broadcasts a single value across groups,
            ``"free"`` samples independently per group, and a ``Pool``
            instance applies non-centered partial pooling across groups.
            Defaults to ``"shared"``.
        by_variable: Same as by_group but along the variable/latent
            axis. Defaults to ``"shared"``.
    """
    prior: dist.Distribution | float
    by_group: Literal["shared", "free"] | Pool = "shared"
    by_variable: Literal["shared", "free"] | Pool = "shared"

    def sample_raw(self, name, n_groups, n_target, trailing=()):
                
        # parameter is a scalar and repeated across groups and variables/latents 
        if not isinstance(self.prior, dist.Distribution):
            deterministic = jnp.asarray(self.prior)
            return numpyro.deterministic(name, deterministic)
        
        # determine shape of parameter and expand + broadcast accordingly
        shape = (1 if self.by_group == "shared" else n_groups, 
                 1 if self.by_variable == "shared" else n_target,
                 *trailing) # trailing supports additional parameter dimensions (e.g., n_predictors in Linear) 
        
        #shape = (n_groups if self.by_group else 1, n_target if self.by_variable else 1)
        random = numpyro.sample(name, self.prior.expand(shape).to_event(len(shape)))
        
        return random
    
    def apply_partial_pooling(self, name, raw, n_groups, n_target, trailing=()):
                
        if not isinstance(self.by_group, Pool) and not isinstance(self.by_variable, Pool):
            return raw
        
        if isinstance(self.by_group, Pool) and isinstance(self.by_variable, Pool):
            raise ValueError("Partial pooling across both groups and variables is not supported")
        
        # either by_group or by_variable are set to pooling at this point
        pool = self.by_group if isinstance(self.by_group, Pool) else self.by_variable
        # hyperparameters are shared along the pooled axis and follow the other axis's setting
        hyper_groups = n_groups if self.by_group == "free" else 1
        hyper_target = n_target if self.by_variable == "free" else 1
        loc = pool.loc(f"{name}_loc", hyper_groups, hyper_target, trailing)
        scale = pool.scale(f"{name}_scale", hyper_groups, hyper_target, trailing)
        unconstrained = loc + scale * raw
        return  pool.transform(unconstrained)
    
    def __call__(self, name, n_groups, n_target, trailing=()):
        
        raw = self.sample_raw(name, n_groups, n_target, trailing)
        coef = self.apply_partial_pooling(name, raw, n_groups, n_target, trailing)
        return jnp.broadcast_to(coef, (n_groups, n_target, *trailing))

@dataclass(frozen=True)
class Pool:
    """Hyperparameters for non-centered partial pooling in ``Param``.

    Attributes:
        loc: ``Param`` for the hyper-mean. Defaults to Normal(0, 1).
        scale: ``Param`` for the hyper-scale. Defaults to HalfNormal(0.5).
            Hyperparameters are always shared along the pooled axis. Along
            the other axis, the defaults follow the pooled ``Param``: one
            value if it is ``"shared"``, one per level if it is ``"free"``.
            Pass a shared ``Param`` to use a single value instead.
        transform: Bijective transform applied after the affine
            ``loc + scale * raw`` step (e.g. ``ExpTransform`` to
            constrain the result to be positive). Defaults to the
            identity transform.
    """
    loc: Param = Param(dist.Normal(0.0, 1.0), by_group="free", by_variable="free")
    scale: Param = Param(dist.HalfNormal(0.5), by_group="free", by_variable="free")
    transform: dist.transforms.Transform = field(default_factory=dist.transforms.IdentityTransform)