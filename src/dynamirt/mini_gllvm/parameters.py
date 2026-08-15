from __future__ import annotations

from typing import Literal
from dataclasses import dataclass, field

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp


@dataclass(frozen=True)
class Param:
    """
    Shape helper. Terms (e.g. GPs) may extend along groups and/or variables/latents.
    These terms may contain parameters (this class) that might themselves be shared or duplicated along axes.
    This class takes either scalars or distributions and broadcasts them to the specified shapes.
    
    Supports non-centered partial pooling for loc-scale family priors.
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
    
    def apply_partial_pooling(self, name, raw, n_groups, n_target):
                
        if not isinstance(self.by_group, Pool) and not isinstance(self.by_variable, Pool):
            return raw
        
        if isinstance(self.by_group, Pool) and isinstance(self.by_variable, Pool):
            raise ValueError("Partial pooling across both groups and variables is not supported")
        
        # either by_group or by_variable are set to pooling at this point
        pool = self.by_group if isinstance(self.by_group, Pool) else self.by_variable
        loc = pool.loc(f"{name}_loc", n_groups, n_target)
        scale = pool.scale(f"{name}_scale", n_groups, n_target)
        unconstrained = loc + scale * raw
        return  pool.transform(unconstrained)
    
    def __call__(self, name, n_groups, n_target, trailing=()):
        
        raw = self.sample_raw(name, n_groups, n_target)
        coef = self.apply_partial_pooling(name, raw, n_groups, n_target)
        return jnp.broadcast_to(coef, (n_groups, n_target, *trailing))

@dataclass(frozen=True)
class Pool:
    loc: Param = Param(dist.Normal(0.0, 1.0))
    scale: Param = Param(dist.HalfNormal(0.5))
    transform: dist.transforms.Transform = field(default_factory=dist.transforms.IdentityTransform)