from .mini_gllvm import gllvm
from .mini_gllvm import Linear

from .mini_gllvm.loadings import Full, Fixed

from .families import _dichotomous, _polytomous

from functools import partial

from typing import Sequence, Callable, Literal

_DICHOTOMOUS = ["1PL", "2PL", "3PL", "4PL"]
_POLYTOMOUS = ["GRM", "PCM", "GPCM"]

def dynamirt(
    model_type: Literal["1PL", "2PL", "3PL", "4PL", "GRM", "PCM", "GPCM"]="2PL",
    n_latent: int =1,
    loadings: Callable | None =None,
    latent_fn: Sequence[Callable] | None =None,
    DIF: Sequence[Callable] | None =None,
    include_residuals: bool | None =None,
    corr: bool = True,
    model_type_kwargs: dict | None =None
    ) -> Callable:
    
    """
    Build a MIRT model with optionally time evolving latent traits.

    Args:
        model_type: IRT family to use. Dichotomous: "1PL"/"2PL"/"3PL"/"4PL".
            Polytomous: "GRM"/"PCM"/"GPCM".
        n_latent: Dimensionality of the latent space.
        loadings: Loading matrix factory (e.g. `Fixed()`, `Unconstrained()`).
            Defaults to `Unconstrained()`, except for "1PL"/"PCM" where it's
            forced to `Fixed()`.
        latent_fn: Latent regression terms (e.g. covariate effects on theta).
            If omitted, a i.i.d residual latent term is added by default (see `include_residuals`).
        DIF: Terms added to the full-rank (item-level) regression to model
            differential item functioning.
        include_residuals: Whether to add a residual latent term. Defaults
            to True only when `latent_fn` is not provided.
        corr: If including residuals, whether to estimate correlations
            between latent variables' residuals.
        model_type_kwargs: Extra keyword arguments passed to the family
            constructor for `model_type`.

    Returns:
        Callable: A numpyro model function.
    """
    
    # ----------------- validation and setting defaults -----------------
    if model_type in ["1PL", "PCM"]:
        if loadings is not None:
            raise ValueError("1PL/PCM fixes the loadings. drop `loadings` or use 2PL/GPCM.")
        loadings = Fixed()
    
    loadings = Full() if loadings is None else loadings
    
    latent_contribution = [] if latent_fn is None else list(latent_fn)
    DIF = [] if DIF is None else list(DIF)
    model_type_kwargs = {} if model_type_kwargs is None else model_type_kwargs
    
    # by default do not include residuals when latent_fn is explicitly modeled.
    if include_residuals is None:
        include_residuals = not latent_contribution

    if not include_residuals and not latent_contribution:
        raise ValueError("No latent terms: set include_residuals=True or pass latent_fn.")

    # ----------------- model construction -----------------
    if include_residuals:
        # estimate correlation between latents by default
        latent_contribution.append(Linear("residuals", predictors="one_", group_by="row_", corr="variables" if corr else None))    
        
    if model_type in _DICHOTOMOUS:
        # generate item intercept for dichotomous models
        full_rank = [Linear("item_intercept", predictors="one_"), *DIF]
        family_fn = _dichotomous(model_type, **model_type_kwargs)
    elif model_type in _POLYTOMOUS:
        # polytomous models create their own intercepts in their family_fn
        full_rank = [*DIF]
        family_fn = _polytomous(model_type, **model_type_kwargs)
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