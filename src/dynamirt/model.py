from .mini_gllvm import gllvm
from .mini_gllvm import Linear

from .mini_gllvm.families import Bernoulli
from .mini_gllvm.loadings import Unconstrained, Fixed

from .families import _irt_NPL

from functools import partial

from typing import Sequence, Callable, Literal

def dynamirt(
    model_type: Literal["1PL", "2PL", "3PL", "4PL"]="2PL",
    n_latent: int =1,
    loadings: Callable | None =None,
    latent_fn: Sequence[Callable] | None =None,
    DIF: Sequence[Callable] | None =None,
    include_residuals: bool | None =None,
    model_type_kwargs: dict | None =None
):
    if model_type == "1PL":
        if loadings is not None:
            raise ValueError("1PL fixes the loadings. drop `loadings` or use 2PL.")
        loadings = Fixed()
    
    loadings = Unconstrained() if loadings is None else loadings
    
    latent_contribution = [] if latent_fn is None else list(latent_fn)
    DIF = [] if DIF is None else list(DIF)
    model_type_kwargs = {} if model_type_kwargs is None else model_type_kwargs
    
    # by default do not include residuals when latent_fn is explicitly modeled.
    if include_residuals is None:
        include_residuals = not latent_contribution

    if not include_residuals and not latent_contribution:
        raise ValueError("No latent terms: set include_residuals=True or pass latent_fn.")
        
    if include_residuals:
        latent_contribution.append(Linear("residuals", predictors="one_", group_by="row_"))    
        
    full_rank = [Linear("item_intercept", predictors="one_"), *DIF]
    
    if model_type in ["1PL", "2PL"]:
        family_fn = Bernoulli()
    elif model_type in ["3PL", "4PL"]:
        family_fn = _irt_NPL(model_type, **model_type_kwargs)
    else:
        raise ValueError(f"Unkown model type {model_type}")
    
    model = partial(
        gllvm,
        n_latent=n_latent,
        full_rank_regression=full_rank,
        latent_regression=latent_contribution,
        loadings=loadings,
        family=family_fn,
        latent_site_name="theta"
    )
    
    return model