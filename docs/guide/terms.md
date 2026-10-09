# Covariates & Built-In Terms

In **dynamirt** you can freely specify the functional form of the latent construct using the `latent_terms` argument. The contribution the terms are added together and the result is available in the posterior trace as `"theta"`. The contribution of each individual term is available through `"latent.<term_name>.contribution"` `(n_obs, n_latent)`. Because these terms operate on the response variables only by proxy through a lower dimensional representation, this is also referred to as low-rank regression. The package contains multiple built-ins that are intended to simplify working with time series data specifically.

The available built-in terms are:

| Term | Description |
|---|---|
| {py:class}`Linear <dynamirt.Linear>` | Intercepts and linear effects |
| {py:class}`GRW <dynamirt.GRW>` | Gaussian random walks on equally spaced steps |
| {py:class}`AR1 <dynamirt.AR1>` | Stationary autoregressive processes on equally spaced steps |
| {py:class}`GP <dynamirt.GP>` | Exact Gaussian processes |
| {py:class}`HSGP <dynamirt.HSGP>` | Basis approximations to Gaussian processes |

On top of adding terms sequentially to `latent_terms`, you will need to supply a `covariates` dictionary that provides all covariates that you refer to in the model construction such as respondent ids and time. For all built-in terms, entries in the covariate dict need to be either numpy arrays of shape `(n_obs, )` or scalars. There are two default covariates that are added internally and that you can refer to in the model construction whether you pass a covariate dictionary or not. The first is `"one_"` which contains the constant 1.0 (shape `(1,)`, broadcast over observations). This can be used to to create intercepts and is the default predictor for {py:class}`Linear <dynamirt.Linear>` if no custom one is passed. The second one is `"obs_"` which contains the row numbers of the responses (shape `(n_obs,)`). For each covariate used in `group_by` or discrete `order_by` (in GRW, AR1), the total number of categories must be supplied through `index_sizes`.

The following builds and fits a model that includes a respondent-specific intercept and linear trend over time. The default prior for the linear coefficient is `Normal(0, 1)`:

```python
import numpy as np
from dynamirt import Linear

covariates = {
    "respondent_id": np.repeat(np.arange(9), 3), # 9 respondents
    "time_point": np.tile(np.arange(3), 9)       # 3 time points per respondent
}

model = dynamirt(
    model_type="GRM", 
    n_latent=2, 
    loadings=Confirmatory(Q, positive),
    model_type_kwargs={"n_cat": 5},
    latent_terms=[
        Linear("intercept", group_by="respondent_id"),
        Linear("slope", predictors="time_point", group_by="respondent_id")
        ],
    index_sizes={"respondent_id": 9}
    )

fit = fit_mcmc(model, responses, covariates)
idata = fit.to_idata()
```

If no latent terms are passed, dynamirt supplies a default {py:class}`Linear("residuals", predictors="one_", group_by="obs_") <dynamirt.Linear>` which adds i.i.d `Normal(0, 1)` noise to all observations for each latent dimension. In the absence of any other latent predictors this is simply standard static IRT.

## Differential Item Functioning

Terms that directly operate on the response variables (i.e., full-rank regression) enable differential item functioning (DIF). They can be specified through the `item_terms` argument in {py:func}`dynamirt() <dynamirt.dynamirt>`. DIF might require additional restrictions such as anchor items to prevent translation identification issues where a constant shift in location across all items can equally be expressed as a shift in the latents. Suitable priors may provide soft identification in the absence of anchors. More documentation to be added.

[Building Models](models.md) · [Custom Terms](custom_terms.md) · [Priors, Parameters & Multilevel Models](parameters.md) · [Dynamic tutorial](../tutorials/tutorial_dynamic.ipynb)
