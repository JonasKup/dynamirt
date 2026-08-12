from dataclasses import dataclass
from typing import Mapping

from jax.typing import ArrayLike

# minimal context dataclass passed to subfunctions
@dataclass(frozen=True, eq=False)
class _Context:
    
    responses: ArrayLike
    covariates: Mapping[str, ArrayLike]
    n_obs: int
    n_var: int
    n_latent: int