from .mini_gllvm import gllvm
from .mini_gllvm import Linear

from .mini_gllvm.families import Bernoulli
from .mini_gllvm.loadings import Unconstrained

from functools import partial

from typing import Sequence, Callable

def dynamirt(
    model_type: str="2PL",
    n_latent: int = 1,
    loadings: Callable | None =None,
    latent_fn: Sequence[Callable] | None=None,
    DIF: Sequence[Callable] | None=None,
    include_residuals: bool | None=None
):
    
    loadings = Unconstrained() if loadings is None else loadings
    latent_contribution = [] if latent_fn is None else list(latent_fn)
    DIF = [] if DIF is None else list(DIF)

    # by default do not include residuals when latent_fn is explicitly modeled.
    if include_residuals is None:
        include_residuals = not latent_contribution

    if not include_residuals and not latent_contribution:
        raise ValueError("No latent terms: set include_residuals=True or pass latent_fn.")
        
    if include_residuals:
        latent_contribution.append(Linear("residuals", predictors="one_"))    
        
    full_rank = [Linear("item_intercept", predictors="one_"), *DIF]
    
    model = partial(
        gllvm,
        n_latent=n_latent,
        full_rank_regression=full_rank,
        latent_regression=latent_contribution,
        loadings=loadings,
        family=Bernoulli(),
        latent_site_name="theta"
    )
    
    return model