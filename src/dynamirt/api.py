from .gllvm.model import gllvm
from .gllvm.terms.linear import Linear

from .gllvm.loadings import Full, Fixed

from ._families import _dichotomous, _polytomous

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
        model_type: IRT response family. Dichotomous options:
            ``"1PL"``, ``"2PL"``, ``"3PL"``, ``"4PL"``. Polytomous
            options: ``"GRM"`` (graded response model),
            ``"PCM"`` (partial credit model),
            ``"GPCM"`` (generalised partial credit model).
            Defaults to ``"2PL"``.
        n_latent: Dimensionality of the latent trait space. Defaults
            to 1.
        loadings: Loading-matrix factory (e.g. ``Fixed()``,
            ``Full()``, ``Confirmatory(Q)``). Defaults to ``Full()``
            (unconstrained), except for ``"1PL"`` and ``"PCM"`` where
            it is forced to ``Fixed()``.
        latent_fn: Sequence of latent-regression term callables (e.g.
            time-series terms, covariate effects on theta). If None
            and ``include_residuals`` is not explicitly False, an
            i.i.d. residual term is added automatically.
        DIF: Sequence of full-rank regression term callables added
            alongside the item intercepts to model differential item
            functioning. Defaults to None.
        include_residuals: Whether to append an i.i.d. residual latent
            term. Defaults to True when ``latent_fn`` is None,
            False otherwise.
        corr: If True and residuals are included, an LKJ-Cholesky
            correlation structure is estimated across latent
            dimensions. Defaults to True.
        model_type_kwargs: Extra keyword arguments forwarded to the
            IRT family constructor for `model_type`.
        
    Returns:
        A numpyro function with signature ``model(responses, covariates, ...)``
        ready for use with ``fit_mcmc`` or ``fit_svi``.
        
    Raises:
        ValueError: If ``loadings`` is provided for ``"1PL"`` or
            ``"PCM"`` (these fix all loadings to 1).
        ValueError: If no latent terms would be active (both
            ``latent_fn`` and ``include_residuals`` are empty/False).
        ValueError: If ``model_type`` is not recognised.
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