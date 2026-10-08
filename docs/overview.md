# Overview


## Building Models

First and foremost, dynamirt is a model builder and the main entry point is the `dynamirt()` function. The simplest model you can build is a static unidimensional one, where all items can be assumed to load onto the latent in the same direction. `fit_mcmc` fits it and `.to_idata` turns it into an [ArviZ](https://www.arviz.org/en/latest/) style [xarray](https://docs.xarray.dev/en/stable/) `DataTree` for easy diagnostics:

```python
from dynamirt import dynamirt, fit_mcmc

model = dynamirt(model_type="2PL")
fit = fit_mcmc(model, responses)
idata = fit.to_idata()
```

Here, `responses` is your data, an `(n_obs, n_items)` array. Binary items are coded 0/1,
ordinal items use categories `0, ..., n_cat - 1` and missing responses are `np.nan`.
The responses are assumed to be in [long format](https://en.wikipedia.org/wiki/Wide_and_narobs_data)
with one row per observation occasion and one column per item.

For binary responses dynamirt supports the `1PL`, `2PL`, `3PL`, and `4PL` models. For ordinal responses, the graded response model (`GRM`), partial credit model (`PCM`) and generalized PCM (`GPCM`) are supported.

The latent construct is called `"theta"` `(n_obs, n_latent)` in the trace. The item intercepts for dichotomous models appear under the name `"measurement.intercept"` `(n_items,)`.

### Multidimensional IRT

If a single shared latent is not a sufficient assumption for your data, you have to start thinking about the `loading` structure. Essentially, this is what tells your model how each item maps onto the latent space. 

One common assumption is that the loadings structure is known beforehand. This is supported through the `Confirmatory(Q)` loadings factory function. Q is an `(n_items, n_latent)` matrix with 0/1 entries that tells the model exactly which item maps onto which latent factor. The package ships with a small helper to build it from a list of items. As an example, let's assume items that map onto two seperate latents (positive and negative affect) and a simple loading structure (i.e., each item only loads onto one latent):

```python
from dynamirt import q_matrix

ITEMS = ["cheerful", "energetic", "interested", "relaxed", "anxious", "irritable", "down", "bored"]

# 1) using item labels
Q = q_matrix([
    ["cheerful", "energetic", "interested", "relaxed"], 
    ["anxious", "irritable", "down", "bored"]
    ], items=ITEMS)

# 2) using item indices
Q = q_matrix([[0, 1, 2, 3], [4, 5, 6, 7]], items=8)
```

To fit this, we do pretty much the same as above, now making use of the `n_latent` and `loadings` arguments. Here, we assume that all items are coded in the same direction, i.e., a 1 response means this item has been endorsed. The likelihood does not understand the difference between all positive and all negative loadings though. For this reason we use a `positive` mask `(n_items, n_latent)` to choose two anchor items (one per latent) that we constrain positive. In practice this means replacing the default `Normal(0, 1)` priors with `LogNormal(0, 0.5)` on just the anchor items. Generally, this should be enough to break the invariance. The priors can be adjusted using the `free_prior` and `positive_prior` keywords on `Confirmatory()`. We also supply custom names for the latents and items to `.to_idata` which will then be used in all subsequent plots.

```python
from dynamirt import dynamirt, fit_mcmc, Confirmatory

positive = q_matrix([["cheerful"], ["anxious"]], items=ITEMS)

model = dynamirt(model_type="2PL", n_latent=2, loadings=Confirmatory(Q, positive))
fit = fit_mcmc(model, responses)
idata = fit.to_idata(coords={"latent": ["positive_affect", "negative_affect"], "item": ITEMS})
```

Feel free to take a look at other supported loading structures. In most cases you will likely want to stick with `Confirmatory()`. Through the `Sparsity()` factory, dynamirt also includes experimental support for exploratory set ups by assuming a sparse loading structure with regularized horseshoe priors. This encourages a sparse orientation but does not guarantee identification. The results still suffer from sign and permutation invariances.

The loading structure itself will appear in the trace (e.g., in the `idata` object) under the name `"loadings"` `(n_items, n_latent)`. All related parameters will have the `"loadings."` prefix.

When you are sticking with the default parametrization of the latent construct as in all the examples above (e.g., no custom time series structure), you can also estimate the correlation between latents by setting `corr=True` in `dynamirt()`. The cholesky decomposition of the correlation matrix will show up as `"latent.residuals.correlation_cholesky"` `(n_latent, n_latent)` in the trace when `n_latent > 1`.

### Dichotomous Measurement Models

The above examples all make use of the two parameter logistic (2PL) model. All other binary measurement models work just the same and you can simply replace the `model_type` string in `dynamirt()` to your preferred measurement model. 

The 1PL model forces the loading structure to `Fixed()` and will throw an error if you supply one manually. The 3PL and 4PL models can be weakly identified especially if you have little data.

### Polytomous Measurement Models

You can build ordinal response models by changing the `model_type` to `GRM`, `PCM`, or `GPCM`. On top of that you have to supply the number of categories for each item.

If all items have the same number of categories (e.g., five):

```python
model = dynamirt(
    model_type="GRM", 
    n_latent=2, 
    loadings=Confirmatory(Q, positive),
    model_type_kwargs={"n_cat": 5})
```

If items have different number of categories you have to supply them all in item-column order. As an example, let's say positive affect items have four categories while negative affect items have five:

```python
model = dynamirt(
    model_type="GRM", 
    n_latent=2, 
    loadings=Confirmatory(Q, positive),
    model_type_kwargs={"n_cat": [4, 4, 4, 4, 5, 5, 5, 5]})
```

The `model_type_kwargs` are passed to the measurement model factory and also let you modify the default priors.

### Covariates & Built-In Terms

In **dynamirt** you can freely specify the functional form of the latent construct using the `latent_terms` argument. The contribution the terms are added together and the result is available in the posterior trace as `"theta"`. The contribution of each individual term is available through `"latent.<term_name>.contribution"` `(n_obs, n_latent)`. Because these terms operate on the response variables only by proxy through a lower dimensional representation, this is also referred to as low-rank regression. The package contains multiple built-ins that are intended to simplify working with time series data specifically.

On top of adding terms sequentially to `latent_terms`, you will need to supply a `covariates` dictionary that provides all covariates that you refer to in the model construction such as respondent ids and time. For all built-in terms, entries in the covariate dict need to be either numpy arrays of shape `(n_obs, )` or scalars. There are two default covariates that are added internally and that you can refer to in the model construction whether you pass a covariate dictionary or not. The first is `"one_"` which contains the constant 1.0 (shape `(1,)`, broadcast over observations). This can be used to to create intercepts and is the default predictor for `Linear` if no custom one is passed. The second one is `"obs_"` which contains the row numbers of the responses (shape `(n_obs,)`). For each covariate used in `group_by` or discrete `order_by` (in GRW, AR1), the total number of categories must be supplied through `index_sizes`.

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

If no latent terms are passed, dynamirt supplies a default `Linear("residuals", predictors="one_", group_by="obs_")` which adds i.i.d `Normal(0, 1)` noise to all observations for each latent dimension. In the absence of any other latent predictors this is simply standard static IRT.

#### Differential Item Functioning

Terms that directly operate on the response variables (i.e., full-rank regression) enable differential item functioning (DIF). They can be specified through the `item_terms` argument in `dynamirt()`. DIF might require additional restrictions such as anchor items to prevent translation identification issues where a constant shift in location across all items can equally be expressed as a shift in the latents. Suitable priors might provide soft identification in the absence of anchors. More documentation to be added.

### Custom Terms

The available built-in terms are:

| Term | Description |
|---|---|
| `Linear` | Intercepts and linear effects |
| `GRW` | Gaussian random walks on equally spaced steps |
| `AR1` | Stationary autoregressive processes on equally spaced steps |
| `GP` | Exact Gaussian processes |
| `HSGP` | Basis approximations to Gaussian processes |

While the built-ins lean toward time series modeling, they don't exhaustively cover all longitudinal models you may want to build and maybe you want to build something else entirely (e.g., a spatial model). 

Creating custom terms is relatively straight forward because you just need to write a NumPyro/JAX function that implements your desired operation. The most important requirement for custom term functions is that they return their result as an array of shape `(n_obs, n_target)`. `n_target` should be the latent dimension if used as a `latent_term` or the number of items if used as an `item_term`. 

All custom term functions must take two arguments: a dynamirt `ModelContext` and `n_target`. The ModelContext lets you access the `responses` array, the `covariates` dict, as well as `n_obs`, `n_var` (number of items), `n_latent`, and `index_sizes`.

As an example let's create a GPMHT style model ([McBride & Wang, 2026](https://doi.org/10.1017/psy.2026.10082)) in which respondents share a B-spline mean function over time and HSGPs describe respondent specific deviations. dynamirt does not have built-in support for splines so we create a custom term. We precompute the spline basis and supply it to the `spline_mean` function through the `covariates` dict. Finally, we wrap it in `CustomTerm()` and hand it to the model constructor together with the built-in HSGPs:

```python
from scipy.interpolate import BSpline
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from dynamirt import dynamirt, CustomTerm, HSGP, Param

knots = np.r_[
    np.repeat(0.0, 4),
    np.linspace(0.0, 1.0, 15)[1:-1],
    np.repeat(1.0, 4),
]
covariates["time"] = time  # scaled to [0, 1] for this spline basis
covariates["spline_basis"] = (
    BSpline.design_matrix(time, knots, k=3).toarray()[:, 1:]
)

def spline_mean(ctx, n_targets):
    B = jnp.asarray(ctx.covariates["spline_basis"])
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0).expand((n_targets,)).to_event(1))
    z = numpyro.sample("z", dist.Normal(0, 1).expand((B.shape[1], n_targets)).to_event(2))
    beta = numpyro.deterministic("beta", z * sigma)
    return B @ beta

model = dynamirt(
    model_type="GRM",
    latent_terms=[
        CustomTerm("mean_curve", spline_mean),
        HSGP(
            "trajectories", "time", group_by="respondent_id",
            kernel="Matern", nu=1.5, ell=2.341, m=31,
            length=Param(dist.LogNormal(0, 0.5)),
        ),
    ],
    model_type_kwargs={"n_cat": n_cat},
    index_sizes={"respondent_id": n_respondents}
)
```

For an additional example where the latent space is parametrized by a neural differential equation refer to this notebook.

## Priors, Parameters & Multilevel Models

All default priors in dynamirt can be overwritten. Built-in terms expose them in their constructors, measurement models expose them through `model_type_kwargs` in `dynamirt()`. Due to the nature of latent variable models (regressing on multiple response variables simultaneously) the shape of these can become unclear quickly. This is why dynamirt includes a shallow `Param` abstraction that lets you implement custom priors and specify parameter shapes by keyword. Together with `Pool` this also enables single-level hierarchical models without needing any bespoke NumPyro code.

Here is an example showing the usage of `Param` and `Pool` to build a simple multilevel model with group- and population-level intercepts and a shared slope: 

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

Multiple things are happening here. We use `Param` to set the prior on the raw linear coeffienct to Normal(0, 2) (default is Normal(0, 1)); `Pool` then shifts and scales these draws. Each parameter that is specified through Param has two axes it can be expanded along. `by_group` refers to the grouping axis, e.g. respondent_id. `by_variable` refers to either the latent dimension or the number of items, depending on whether this parameter's parent term is used in `latent_terms` or `item_terms`. Both `by_group` and `by_variable` can either be "free" (expanded along this axis), "shared" (single value that is broadcast along this axis) or `Pool()` (non-centered hierarchical pooling).

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

For `Param`, the default is "shared" on both `by_group` and `by_variable`. The above will create one HSGP trajectory per respondent and latent dimension because of the `group_by` in HSGP. However, there will be a single length scale parameter (shape `(1, 1)`) shared between respondents (`by_group = "shared"`) and latents (`by_variable = "shared"`).

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

This last example introduces non-centered hierarchical pooling on the the respondent axis. In total there will be `n_latent * n_respondent + 2 * n_latent` sampled parameters: raw draws `(n_respondent, n_latent)` and a population-level mean and scale `(1, n_latent)` each. The exponential transform keeps the resulting length scales positive. The default priors for the hyper mean and scale are `Normal(0,1)` and `HalfNormal(0.5)` respectively. They can be adjusted through `Pool()`.

Pooling across latent dimensions with relatively few dimensions seems hardly necessary at first, however, if used in `item_terms` instead, it can be used to introduce shrinkage over the items axis which might be desirable for differential item functioning.

## Fitting Models

Models can be fit either through `fit_mcmc(model, responses, covariates, ...)` for Markov chain Monte Carlo or `fit_svi(model, responses, covariates, ...)` for stochastic variational inference. Both return a `FitResult` object that contains the original NumPyro MCMC and SVI machinery in `result.inference`. It also provides a unified `.to_idata` function for both SVI and MCMC inference that creates ArviZ compatible inference data.

The output of the `dynamirt()` model constructor is simply a NumPyro model so you don't need to make use of the above helpers if you prefer to stick directly with NumPyro for inference. The only drawback to that is that the built-in helpers set some dimension names that subsequent dynamirt plotting functions make use of.

## Plots

### Loadings

The estimated loading structure can be visualized as a heatmap using `plot_loadings`. In the below example, no cross-loadings will be visible because we specify a simple structure in the Q matrix:

```python
from dynamirt import dynamirt, fit_mcmc, plot_loadings

ITEMS = ["cheerful", "energetic", "interested", "relaxed", "anxious", "irritable", "down", "bored"]

Q = q_matrix([[0, 1, 2, 3], [4, 5, 6, 7]], items=8)

model = dynamirt(model_type="2PL", n_latent=2, loadings=Confirmatory(Q, positive))
fit = fit_mcmc(model, responses)
idata = fit.to_idata(coords={"latent": ["positive_affect", "negative_affect"], "item": ITEMS})

plot_loadings(idata)
```

### Item Characteristic Curves

The package can automatically visualize the estimated item characteristic curves independently of the selected measurement model. Internally, this is done by replacing the latent terms with a regular grid in the posterior and tracing across it. 

This example plots the items characteristic curves for the four items on the positive affect latent on a single matplotlib axis:

```python
import matplotlib.pyplot as plt
from dynamirt import item_curves, plot_item_curves

icc = item_curves(fit, posterior=idata["posterior"], latent=["positive_affect"])

fig, ax = plt.subplots()

for item in ["cheerful", "energetic", "interested", "relaxed"]:
    plot_item_curves(icc, item=item, ax=ax)

```
The example above considers only the positive affect latent conditioned on negative_affect == 0. This is not an issue here because we specified a simple loading structure (each item loads either onto the positive or negative affect latent). For a two-dimensional model that also includes some cross loadings, we can visualize the decision boundaries in the latent space instead. The `item_curves` function varies up to two latents, holding any remaining dimensions at zero.

```python
icc = item_curves(fit, posterior=idata["posterior"], latent=["positive_affect", "negative_affect"])
plot_item_curves(icc, item="cheerful")
```

Documentation on item curves in the presence of differential item functioning to be added.

### Latent trajectories

Because dynamirt is geared toward time series modeling, it also includes a helper to quickly plot latent trajectories over time for a single respondent. This is done through the `plot_trajectories` function. You need to specify the name of the covariate that responds to the time dimensions, as well as filter down the posterior to a single respondent through the `coords` keyword which will be a familiar workflow to ArviZ users.

```python
import numpy as np
from dynamirt import Linear, plot_trajectories

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
        Linear("intercept", group_by="respondent_id"),
        Linear("slope", predictors="time_point", group_by="respondent_id")
        ],
    index_sizes={"respondent_id": 9}
    )

fit = fit_mcmc(model, responses, covariates)
idata = fit.to_idata(coords={"latent": ["positive_affect", "negative_affect"]})

plot_trajectories(idata, time="time_point", coords={"respondent_id": 0}, latent="positive_affect")
```

By default this makes use of `"theta"`, i.e., the combined contributions of all latent terms. You can filter this down to a specific term by using the `var_name` keyword. The below only plots the respondent's slope while ignoring the intercept:

```python
plot_trajectories(idata, time="time_point", var_name="latent.slope.contribution", coords={"respondent_id": 0}, latent="positive_affect")
```

Refer to the guide on scoped parameter names to understand parameter naming schemes.

## Scoped Parameter Names

Depending on how the model was specified, the parameter names that end up in the posterior trace can vary quite a bit which might be confusing when looking for a specific quantity of interest. This is a small guide to help identify the correct parameter names.

On the top level there are two reserved names recorded by every model that was constructed using `dynamirt()`. The first is `"theta"` which is the parameter name for the full latent construct (e.g., the sum of all latent terms supplied to the `latent_terms` argument of `dynamirt()`). The second is `"loadings"` which is the full loading structure. Below that, parameters are scoped by whichever aspect of the model they belong to. Deterministic sites are omitted when fitting with `return_deterministic=False`. Shapes below exclude posterior sampling dimensions.

### Loadings

All parameters related to the loading structure are scoped under `"loadings."`. E.g., for a loading structure specified through `Confirmatory()`, the full matrix will be in `"loadings"`. Additionally, the posterior trace will contain separate parameter recordings for the unconstrained (`"loadings.confirmatory_free"`) and positive constrained (`"loadings.confirmatory_positive"`) item loadings.

Site names below omit the `loadings.` prefix.

| Factory | Sites (shape) |
|---|---|
| `Fixed` | No additional sites |
| `Full` | `full` `(n_items, n_latent)` |
| `Confirmatory` | `confirmatory_free` `(n_free,)`, `confirmatory_positive` `(n_positive,)` |
| `Sparsity` | `beta`, `lambda`, `lambda_aux` `(n_items, n_latent)`; `tau`, `tau_aux`, `c_sq` scalar |

`n_free` and `n_positive` count the corresponding estimated entries in Q; empty sets create no site. `Sparsity` sites are raw coefficients and shrinkage variables; the assembled matrix is `loadings`.

### Measurements

All parameters related to the measurement model (e.g. 2PL, GRM, etc) are scoped under `"measurement."`. This is why the dichotomous item intercepts can be found in `"measurement.intercept"`. 3PL models have an additional `"measurement.lower_asymptote"` parameter and 4PL models also include the `"measurement.upper_asymptote"`, etc.

Site names below omit the `measurement.` prefix; `K` is the largest category count.

| Model | Sites (shape) |
|---|---|
| `1PL`, `2PL`, `3PL`, `4PL` | `intercept` `(n_items,)` |
| `3PL`, `4PL` | `lower_asymptote` `(n_items,)` |
| `4PL` | `upper_asymptote_fraction`, `upper_asymptote` `(n_items,)` |
| `GRM` | `baseline_probs` `(n_items, K)`, `cutpoints` `(n_items, K - 1)` |
| `PCM`, `GPCM` | `steps` `(n_items, K - 1)` |

With per-item category counts, GRM instead records `baseline_probs_<k>` `(n_items_with_k_categories, k)` for each distinct count. PCM/GPCM additionally record `steps.raw` `(sum(n_cat - 1),)`. The assembled cutpoints/steps are zero-padded. Cutpoints and upper asymptotes are deterministic; steps are deterministic only with per-item counts.

### Latent & Item Terms

Parameters relating to custom and built-in latent terms get the `"latent.<term_name>."` prefix. The  `(n_obs, n_latent)` contribution that each term makes to `"theta"` is available as `"latent.<term_name>.contribution"`. The length scale parameter for an HSGP term called "trajectory" will be called `"latent.trajectory.lengthscale"`.

Similarly, item terms get the `"item.<term_name>."` prefix, with contributions of shape `(n_obs, n_items)`.

Site names below omit the `latent.<term_name>.` or `item.<term_name>.` prefix. `T` is the number of targets (latents or items), `G` the number of groups, and `P` the number of predictors. Shared realizations use `T = 1`.

| Term | Sites (shape) |
|---|---|
| `Linear` | `coef.raw`, `coef` `(G, T, P)`; optional `correlation_cholesky` |
| `GRW` | `scale`; `innovations.raw`, `state` `(n_time, G, T)` |
| `AR1` | `scale`, `phi`; `innovations.raw`, `state` `(n_time, G, T)` |
| `GP` | User-named kernel parameters; `state.raw`, `state` `(n_max_points_per_group, G, T)` |
| `HSGP` | `amplitude`, `lengthscale`; `basis_weights.raw` `(G, T, n_basis)` |

Every term also records `contribution` `(n_obs, T)`. `coef`, `state`, and `contribution` are deterministic. For `Linear`, reference coding reduces the raw group axis to `G - 1`; sharing reduces raw axes to 1. `correlation_cholesky` is square, sized by the correlated axis: `P`, `T`, or `P * T`.

Distribution-valued `Param` sites use `(G, T)` with shared axes reduced to 1 (and a trailing `P` for linear coefficients); fixed scalar parameters have shape `()`. Pooling adds `<parameter>.loc` and `<parameter>.scale`, shared along the pooled axis. For `Linear` these are `coef.loc` and `coef.scale`. Pooled sites contain raw draws; the transformed values are used internally.

## Generalized Linear Latent Variable Models (GLLVM) \[WIP\]

Documentation to be added