from dataclasses import dataclass, field
from typing import Mapping
from numbers import Integral

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
        index_sizes: Fixed axis sizes for integer-coded grouping and time
            covariates. Codes may select any subset of an axis.
    """
    
    responses: ArrayLike | None
    covariates: Mapping[str, ArrayLike]
    n_obs: int
    n_var: int
    n_latent: int
    
    index_sizes: Mapping[str, int] = field(default_factory=dict)

    def index(self, key):
        """Return supplied integer codes and their configured axis size.

        None denotes a single group. Named inputs must be concrete integer
        scalars or vectors of length n_obs, with codes in [0, index_sizes[key]).
        Scalars broadcast to every row; subsets and unused levels are allowed.
        """
        if key is None:
            return np.zeros(self.n_obs, dtype=int), 1
        if key not in self.index_sizes:
            raise ValueError(f"Missing index_sizes entry for {key!r}")
        size = self.index_sizes[key]
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
            raise ValueError(f"index_sizes[{key!r}] must be a positive integer")
        codes = np.asarray(self.covariates[key])
        if not np.issubdtype(codes.dtype, np.integer):
            raise ValueError(f"Index covariate {key!r} must contain integer codes")
        if codes.ndim != 0 and codes.shape != (self.n_obs,):
            raise ValueError(f"Index covariate {key!r} must be scalar or have shape ({self.n_obs},)")
        if np.any((codes < 0) | (codes >= size)):
            raise ValueError(f"Index covariate {key!r} must contain codes in [0, {size})")
        return np.broadcast_to(codes, (self.n_obs,)), int(size)

    def design(self, predictors):
        """Stack named scalar/vector covariates into a JAX design matrix.

        Accepts a name or sequence of names and broadcasts scalars to n_obs.
        Access matrix-valued inputs directly through covariates.
        """

        keys = [predictors] if isinstance(predictors, str) else list(predictors)
        cols = [jnp.broadcast_to(jnp.asarray(self.covariates[k]).ravel(), (self.n_obs,)) for k in keys]
        return jnp.stack(cols, axis=-1)
