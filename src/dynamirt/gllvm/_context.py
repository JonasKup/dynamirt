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
        is_predictive: Whether the model is currently predicting for new data
        train_covariates: Covariates used for fitting the model. 
            Only required when using Predictive with ExactGPs.
        train_obs: Number of observations when fitting the model.
            Only required when using Predictive with ExactGPs.
    """
    
    responses: ArrayLike
    covariates: Mapping[str, ArrayLike]
    n_obs: int
    n_var: int
    n_latent: int
    
    is_predictive: bool = False
    train_covariates: Mapping[str, ArrayLike] | None = None
    train_obs: int | None = None
    
    def _factorize(self, key, use_train=False):
        """interpret covariate as index column for grouping"""
        covariates = self.train_covariates if use_train else self.covariates
        n_obs = self.train_obs if use_train else self.n_obs
        
        if key is None:
            return np.zeros(n_obs, int), 1
        codes, idx = np.unique(np.asarray(covariates[key]).ravel(), return_inverse=True)
        return idx, codes.size

    def _design(self, predictors, use_train=False):
        """stack multiple covariates into single array if multiple were given. 
        Broadcast scalar values to correct length."""
        covariates = self.train_covariates if use_train else self.covariates
        n_obs = self.train_obs if use_train else self.n_obs

        keys = [predictors] if isinstance(predictors, str) else list(predictors)
        cols = [np.broadcast_to(np.asarray(covariates[k]).ravel(), (n_obs,)) for k in keys]
        return np.stack(cols, axis=-1)
