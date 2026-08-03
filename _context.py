from dataclasses import dataclass
from typing import Mapping

from jax.typing import ArrayLike

# minimal context dataclass passed to subfunctions to keep the module maximally extensible
@dataclass(frozen=True, eq=False)
class _Context:
    
    responses: ArrayLike
    covariates: Mapping[str, ArrayLike]
    n_obs: int # could be derived from responses but convenient
    n_var: int # could be derived from responses but convenient
    n_latent: int