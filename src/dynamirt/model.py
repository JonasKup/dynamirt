from .mini_gllvm import gllvm
from .mini_gllvm import Linear

from .mini_gllvm.families import Bernoulli
from .mini_gllvm.loadings import Unconstrained

from functools import partial

from typing import List, Callable

def dynamirt(
    model_type="2PL",
    n_latent: int = 1,
    loadings=Unconstrained,
    latent_dynamics: List[Callable] = None,
    DIF: List[Callable]=[],
    constrained: bool = False
):
    
    latent_contribution = []
    
    if not constrained:
        residuals = Linear(
            "residuals",
            predictors="one_",
        )
        
        latent_contribution += [residuals]
    
    latent_contribution += latent_dynamics
    
    item_intercept = Linear(
        "intercept",
        predictors="one_"
    )
    
    model = partial(
        gllvm,
        n_latent=n_latent,
        full_rank_regression=[item_intercept, *DIF],
        latent_regression=latent_contribution,
        loadings=loadings(),
        family=Bernoulli(),
        latent_site_name="theta"
    )
    
    return model

# Example:
# model = dynamirt(model_type="2PL", n_latent=1, latent_dynamics=[HSGP("trend", "time", group_by="respondent_id")])