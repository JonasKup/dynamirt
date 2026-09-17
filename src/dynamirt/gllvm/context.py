from dataclasses import dataclass
from typing import Mapping

from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np

# minimal context dataclass passed to subfunctions
@dataclass(frozen=True, eq=False)
class ModelContext:
    
    """Data and dimensions passed to terms, families, and loading factories.

    Bundles the data and shape information that terms, families, and loading
    factories need without requiring each to accept the full argument list.
    
    Also what is passed to custom term functions.

    Attributes:
        responses: Observed response matrix of shape (n_obs, n_var). May contain
            NaN for missing values, or be None when sampling responses.
        covariates: Mapping from covariate name to array. Arrays are typically
            of length n_obs but may be scalar (broadcast at use-site).
        n_obs: Number of observation rows.
        n_var: Number of response variables (columns in responses).
        n_latent: Dimensionality of the latent space.
        is_predictive: Whether training covariates were supplied. This does
            not detect NumPyro Predictive calls in general.
        train_covariates: Covariates used for fitting the model. 
            Only required when using Predictive with ExactGPs.
        train_obs: Number of observations when fitting the model.
            Only required when using Predictive with ExactGPs.
    """
    
    responses: ArrayLike | None
    covariates: Mapping[str, ArrayLike]
    n_obs: int
    n_var: int
    n_latent: int
    
    is_predictive: bool = False
    train_covariates: Mapping[str, ArrayLike] | None = None
    train_obs: int | None = None
    
    def factorize(self, key, use_train=False):
        """Return row group indices and the number of sorted unique levels.

        None denotes a single group. Group covariates must be concrete arrays;
        levels are discovered independently for current and training data.
        """
        covariates = self.train_covariates if use_train else self.covariates
        n_obs = self.train_obs if use_train else self.n_obs
        
        if key is None:
            return np.zeros(n_obs, int), 1
        codes, idx = np.unique(np.asarray(covariates[key]).ravel(), return_inverse=True)
        return idx, codes.size

    def design(self, predictors, use_train=False):
        """Stack named scalar/vector covariates into a JAX design matrix.

        Accepts a name or sequence of names and broadcasts scalars to n_obs.
        Access matrix-valued inputs directly through covariates. With
        use_train=True, uses train_covariates and train_obs instead.
        """
        covariates = self.train_covariates if use_train else self.covariates
        n_obs = self.train_obs if use_train else self.n_obs

        keys = [predictors] if isinstance(predictors, str) else list(predictors)
        cols = [jnp.broadcast_to(jnp.asarray(covariates[k]).ravel(), (n_obs,)) for k in keys]
        return jnp.stack(cols, axis=-1)

    # Compatibility for existing terms.
    _factorize = factorize
    _design = design
