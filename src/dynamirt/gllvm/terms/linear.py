from dataclasses import dataclass
from typing import Literal, Sequence

from .._context import _Context
from ..parameters import Param

import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp

# To do: add Custom term?
# remove corr = both path

@dataclass(frozen=True)
class Linear:
    
    """Linear term ``X @ coef`` for a gllvm regression.

    Contributes an array of shape (n_obs, n_target) where n_target is the
    number of response variables (full-rank predictor), the number of
    latent dimensions (reduced-rank predictor).

    Attributes:
        name: Sample-site prefix. The assembled coefficient tensor is
            stored as a deterministic site under this name.
        predictors: Covariate name or sequence of names to regress on.
        group_by: Optional covariate name giving a grouping factor. A
            separate coefficient vector is drawn per level. Defaults to
            None (single level).
        constraint: ``"reference_coding"`` fixes the first group level's
            coefficients to zero. Defaults to None.
        corr: Axis along which to introduce LKJ-Cholesky correlations
            among coefficients. ``"predictors"`` correlates across
            covariates, ``"variables"`` across response variables or
            latent dimensions, ``"both"`` across both jointly. Defaults
            to None.
        coef: ``Param`` controlling the prior and pooling behaviour of
            the raw coefficients. Defaults to Normal(0, 1) sampled
            independently per group and per variable.
    """
    
    name: str
    predictors: str | Sequence[str]
    group_by: str | None = None
    
    # todo: sum to zero
    constraint: Literal[None, "reference_coding"] = None

    corr: Literal[None, "predictors", "variables", "both"] = None
    coef: Param = Param(dist.Normal(0.0, 1.0), by_group="free", by_variable="free")

    def __call__(self, ctx: _Context, n_vars: int):
        # n_vars is either n_var or n_latents depending on whether Linear is called for eta or u regression

        # raise when group_by would become no-op
        if self.group_by is not None and self.coef.by_group == "shared":
            raise ValueError("group_by is set but coef is shared across groups")

        X = ctx._design(self.predictors)
        n_predictors = X.shape[1]

        idx, n_groups = ctx._factorize(self.group_by)
                   
        # parameter per 'stacked glm' or single parameter broadcast over glm stack
        # if not by_variable is "shared" this is another entry point for latent variables that are not subject to the loadings matrix
        n_target = 1 if self.coef.by_variable == "shared" else n_vars

        # sampling one fewer level for reference coding
        n_free = n_groups - 1 if self.constraint == "reference_coding" else n_groups
        
        # sample the raw parameter, then correlate, then apply partial pooling
        coef = self.coef.sample_raw(f"{self.name}_raw", n_free, n_target, (n_predictors,))

        # b = group_by-level, v = variable/latent, o = over
        if self.corr == "predictors":
            L = numpyro.sample(f"{self.name}_L", dist.LKJCholesky(n_predictors, 1))
            coef = jnp.einsum("bvo,po->bvp", coef, L)
        elif self.corr == "variables":
            L = numpyro.sample(f"{self.name}_L", dist.LKJCholesky(n_target, 1))
            coef = jnp.einsum("bvo,wv->bwo", coef, L)
        elif self.corr == "both":
            m = n_target * n_predictors
            L = numpyro.sample(f"{self.name}_L", dist.LKJCholesky(m, 1))
            b = coef.shape[0]
            coef = (coef.reshape(b, m) @ L.T).reshape(b, n_target, n_predictors)
            
        coef = self.coef.apply_partial_pooling(f"{self.name}_raw", coef, n_free, n_target, (n_predictors,))
        coef = jnp.broadcast_to(coef, (n_free, n_target, n_predictors))

        if self.constraint == "reference_coding":
            coef = jnp.pad(coef, ((1, 0), (0, 0), (0, 0)))

        coef = numpyro.deterministic(self.name, coef) # (n_groups, n_target, n_predictors)
        # n = n_obs, o = over, v = variable/latent
        return jnp.einsum("no,nvo->nv", X, coef[idx]) # (n_obs, n_var)
