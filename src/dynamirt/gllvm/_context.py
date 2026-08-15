from dataclasses import dataclass
from typing import Mapping

from jax.typing import ArrayLike
import numpy as np

# minimal context dataclass passed to subfunctions
@dataclass(frozen=True, eq=False)
class _Context:
    
    responses: ArrayLike
    covariates: Mapping[str, ArrayLike]
    n_obs: int
    n_var: int
    n_latent: int
    
    def _factorize(self, key):
        """interpret covariate as index column for grouping"""
        if key is None:
            return np.zeros(self.n_obs, int), 1
        codes, idx = np.unique(np.asarray(self.covariates[key]).ravel(), return_inverse=True)
        return idx, codes.size

    def _design(self, predictors):
        """stack multiple covariates into single array if multiple were given. 
        Broadcast scalar values to correct length."""
        keys = [predictors] if isinstance(predictors, str) else list(predictors)
        cols = [np.broadcast_to(np.asarray(self.covariates[k]).ravel(), (self.n_obs,)) for k in keys]
        return np.stack(cols, axis=-1)
