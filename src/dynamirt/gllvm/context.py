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
    """
    
    responses: ArrayLike | None
    covariates: Mapping[str, ArrayLike]
    n_obs: int
    n_var: int
    n_latent: int
    
    def factorize(self, key):
        """Return row group indices and the number of sorted unique levels.

        None denotes a single group. Group covariates must be concrete arrays.
        """
        
        if key is None:
            return np.zeros(self.n_obs, int), 1
        codes, idx = np.unique(np.asarray(self.covariates[key]).ravel(), return_inverse=True)
        return idx, codes.size

    def design(self, predictors):
        """Stack named scalar/vector covariates into a JAX design matrix.

        Accepts a name or sequence of names and broadcasts scalars to n_obs.
        Access matrix-valued inputs directly through covariates.
        """

        keys = [predictors] if isinstance(predictors, str) else list(predictors)
        cols = [jnp.broadcast_to(jnp.asarray(self.covariates[k]).ravel(), (self.n_obs,)) for k in keys]
        return jnp.stack(cols, axis=-1)
