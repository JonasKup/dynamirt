from dataclasses import dataclass
from typing import Literal, Sequence

from .._context import _Context
from ..parameters import Param

import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp

@dataclass(frozen=True)
class Linear:
    
    """Linear term `X @ coef` for a gllvm regression.

    Contributes (n_obs, n_target) where n_target is number of variables (full rank), number of latents (reduced rank),
    or 1 broadcast over the stack if `varies_over_variables` is False.

    Attributes:
        name: sample-site prefix; the assembled coefficient is stored as a
            deterministic site under this name.
        predictors: covariate key(s) to regress on
        group_by: covariate key giving a grouping factor; a separate coefficient
            is drawn per level. None means a single level.
        constraint: "reference_coding" fixes the first group level to zero.
        pool_over_groups: share hyperparameters across group levels.
        pool_over_variables: share hyperparameters across variables/latents.
        varies_over_variables: draw one coefficient per variable/latent rather
            than a single one broadcast over all of them.
        corr: "predictors" correlates coefficients across covariates, "variables"
            across variables/latents, via an LKJ Cholesky factor. "both" correlates
            across covariates AND variables/latents.
        prior: prior on the raw coefficient (a non-centered offset when pooling).
        loc_prior: prior on the hyper-mean; used only when pooling.
        scale_prior: prior on the hyper-scale; used only when pooling.
    """
    
    name: str
    predictors: str | Sequence[str]
    group_by: str | None = None
    
    # todo: sum to zero
    constraint: Literal[None, "reference_coding"] = None

    corr: Literal[None, "predictors", "variables", "both"] = None
    coef: Param = Param(dist.Normal(0.0, 1.0))

    def __call__(self, ctx: _Context, n_vars: int):
        # n_vars is either n_var or n_latents depending on whether Linear is called for eta or u regression

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
