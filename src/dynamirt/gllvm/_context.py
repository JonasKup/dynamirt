from dataclasses import dataclass
from typing import Mapping

from jax.typing import ArrayLike
import numpy as np

# minimal context dataclass passed to subfunctions
@dataclass(frozen=True, eq=False)
class _Context:
    
    """Internal context object threaded through model subfunctions.

    Bundles the data and shape information that terms, families, and loading
    factories need without requiring each to accept the full argument list.
    
    Also what is passed to custom term functions.

    Attributes:
        responses: Observed response matrix of shape (n_obs, n_var). May contain
            NaN for missing values.
        covariates: Mapping from covariate name to array. Arrays are typically
            of length n_obs but may be scalar (broadcast at use-site).
        n_obs: Number of observation rows.
        n_var: Number of response variables (columns in responses).
        n_latent: Dimensionality of the latent space.
    """
    
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
