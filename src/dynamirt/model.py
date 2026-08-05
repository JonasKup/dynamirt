from .mini_gllvm import gllvm
from .mini_gllvm import Linear

from .mini_gllvm.families import Bernoulli
from .mini_gllvm.loadings import Unconstrained

from functools import partial

def dynamirt(
    n_latent,
    respondent_var: str,
    time_var: str = None,
    model_type="2PL",
    loadings=Unconstrained,
    DIF=[],
    
):
    
    theta = Linear(
        "theta",
        predictors="one_",
    )
    
    item_intercept = Linear(
        "intercept",
        predictors="one_"
    )
    
    model = partial(
        gllvm,
        n_latent=n_latent,
        full_rank_regression=[item_intercept, *DIF],
        latent_regression=[theta],
        loadings=loadings(),
        family=Bernoulli(),
        latent_site_name="theta"
    )
    
    return model