# Priors, Parameters & Multilevel Models

All default priors in dynamirt can be overwritten. Built-in terms expose them in their constructors, measurement models expose them through `model_type_kwargs` in {py:func}`dynamirt() <dynamirt.dynamirt>`. Due to the nature of latent variable models (regressing on multiple response variables simultaneously) the shape of these can become unclear quickly. This is why dynamirt includes a shallow {py:class}`Param <dynamirt.Param>` abstraction that lets you implement custom priors and specify parameter shapes by keyword. Together with {py:class}`Pool <dynamirt.Pool>` this also enables single-level hierarchical models without needing any bespoke NumPyro code.

Here is an example showing the usage of {py:class}`Param <dynamirt.Param>` and {py:class}`Pool <dynamirt.Pool>` to build a simple multilevel model with group- and population-level intercepts and a shared slope: 

```python
import numpyro.distributions as dist
from dynamirt import Param, Pool
import numpy as np
from dynamirt import Linear

covariates = {
    "respondent_id": np.repeat(np.arange(9), 3),
    "time_point": np.tile(np.arange(3), 9)
}

model = dynamirt(
    model_type="GRM", 
    n_latent=2, 
    loadings=Confirmatory(Q, positive),
    model_type_kwargs={"n_cat": 5},
    latent_terms=[
        Linear("intercept", group_by="respondent_id", coef=Param(dist.Normal(0, 2.0), by_group=Pool(), by_variable="free")),
        Linear("slope", predictors="time_point")
        ],
    index_sizes={"respondent_id": 9}
    )

fit = fit_mcmc(model, responses, covariates)
idata = fit.to_idata()
```

Multiple things are happening here. We use {py:class}`Param <dynamirt.Param>` to set the prior on the raw linear coeffienct to Normal(0, 2) (default is Normal(0, 1)); {py:class}`Pool <dynamirt.Pool>` then shifts and scales these draws. Each parameter that is specified through Param has two axes it can be expanded along. `by_group` refers to the grouping axis, e.g. respondent_id. `by_variable` refers to either the latent dimension or the number of items, depending on whether this parameter's parent term is used in `latent_terms` or `item_terms`. Both `by_group` and `by_variable` can either be "free" (expanded along this axis), "shared" (single value that is broadcast along this axis) or {py:class}`Pool() <dynamirt.Pool>` (non-centered hierarchical pooling).

To showcase why this might be useful here are some examples using a Hilbert space approximate Gaussian process term:

```python
HSGP(
    name="trajectories",
    predictors="time", 
    group_by="respondent_id",
    kernel="Matern", nu=1.5, ell=1.5, m=30,
    length=Param(dist.LogNormal(0, 0.5), by_group="shared", by_variable="shared"),
)
```

For {py:class}`Param <dynamirt.Param>`, the default is "shared" on both `by_group` and `by_variable`. The above will create one HSGP trajectory per respondent and latent dimension because of the `group_by` in HSGP. However, there will be a single length scale parameter (shape `(1, 1)`) shared between respondents (`by_group = "shared"`) and latents (`by_variable = "shared"`).

```python
HSGP(
    name="trajectories",
    predictors="time", 
    group_by="respondent_id",
    kernel="Matern", nu=1.5, ell=1.5, m=30,
    length=Param(dist.LogNormal(0, 0.5), by_group="shared", by_variable="free"),
)
```

The above will let the length scale vary across latents. For an n-dimensional IRT model this means that there is a parameter for each latent dimension, so `n_latent` length scale parameters in total (shape `(1, n_latent)`).

```python
HSGP(
    name="trajectories",
    predictors="time", 
    group_by="respondent_id",
    kernel="Matern", nu=1.5, ell=1.5, m=30,
    length=Param(dist.LogNormal(0, 0.5), by_group="free", by_variable="free"),
)
```

The above lets the length scale vary across latents AND respondents. In total there will be `n_latent * n_respondent` length scale parameters (shape `(n_respondent, n_latent)`).

```python
HSGP(
    name="trajectories",
    predictors="time", 
    group_by="respondent_id",
    kernel="Matern", nu=1.5, ell=1.5, m=30,
    length=Param(dist.Normal(0, 1), by_group=Pool(transform=dist.transforms.ExpTransform()), by_variable="free"),
)
```

This last example introduces non-centered hierarchical pooling on the the respondent axis. In total there will be `n_latent * n_respondent + 2 * n_latent` sampled parameters: raw draws `(n_respondent, n_latent)` and a population-level mean and scale `(1, n_latent)` each. The exponential transform keeps the resulting length scales positive. The default priors for the hyper mean and scale are `Normal(0,1)` and `HalfNormal(0.5)` respectively. They can be adjusted through {py:class}`Pool() <dynamirt.Pool>`.

Pooling across latent dimensions with relatively few dimensions seems hardly necessary at first, however, if used in `item_terms` instead, it can be used to introduce shrinkage over the items axis which might be desirable for differential item functioning.

[Covariates & Built-In Terms](terms.md) · [Custom Terms](custom_terms.md)
