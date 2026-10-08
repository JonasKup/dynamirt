# Building Models

First and foremost, dynamirt is a model builder and the main entry point is the {py:func}`dynamirt() <dynamirt.dynamirt>` function. The simplest model you can build is a static unidimensional one, where all items can be assumed to load onto the latent in the same direction. {py:func}`fit_mcmc <dynamirt.fit_mcmc>` fits it and {py:meth}`.to_idata <dynamirt.fit.FitResult.to_idata>` turns it into an [ArviZ](https://www.arviz.org/en/latest/) style [xarray](https://docs.xarray.dev/en/stable/) `DataTree` for easy diagnostics:

```python
from dynamirt import dynamirt, fit_mcmc

model = dynamirt(model_type="2PL")
fit = fit_mcmc(model, responses)
idata = fit.to_idata()
```

Here, `responses` is your data, an `(n_obs, n_items)` array. Binary items are coded 0/1,
ordinal items use categories `0, ..., n_cat - 1` and missing responses are `np.nan`.
The responses are assumed to be in [long format](https://en.wikipedia.org/wiki/Wide_and_narrow_data)
with one row per observation occasion and one column per item.

For binary responses dynamirt supports the `1PL`, `2PL`, `3PL`, and `4PL` models. For ordinal responses, the graded response model (`GRM`), partial credit model (`PCM`) and generalized PCM (`GPCM`) are supported.

The latent construct is called `"theta"` `(n_obs, n_latent)` in the trace. The item intercepts for dichotomous models appear under the name `"measurement.intercept"` `(n_items,)`.

## Multidimensional IRT

If a single shared latent is not a sufficient assumption for your data, you have to start thinking about the `loading` structure. Essentially, this is what tells your model how each item maps onto the latent space. 

One common assumption is that the loadings structure is known beforehand. This is supported through the {py:func}`Confirmatory(Q) <dynamirt.Confirmatory>` loadings factory function. Q is an `(n_items, n_latent)` matrix with 0/1 entries that tells the model exactly which item maps onto which latent factor. The package ships with a small helper to build it from a list of items. As an example, let's assume items that map onto two seperate latents (positive and negative affect) and a simple loading structure (i.e., each item only loads onto one latent):

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

To fit this, we do pretty much the same as above, now making use of the `n_latent` and `loadings` arguments. Here, we assume that all items are coded in the same direction, i.e., a 1 response means this item has been endorsed. The likelihood does not understand the difference between all positive and all negative loadings though. For this reason we use a `positive` mask `(n_items, n_latent)` to choose two anchor items (one per latent) that we constrain positive. In practice this means replacing the default `Normal(0, 1)` priors with `LogNormal(0, 0.5)` on just the anchor items. Generally, this should be enough to break the invariance. The priors can be adjusted using the `free_prior` and `positive_prior` keywords on {py:func}`Confirmatory() <dynamirt.Confirmatory>`. We also supply custom names for the latents and items to {py:meth}`.to_idata <dynamirt.fit.FitResult.to_idata>` which will then be used in all subsequent plots.

```python
from dynamirt import dynamirt, fit_mcmc, Confirmatory

positive = q_matrix([["cheerful"], ["anxious"]], items=ITEMS)

model = dynamirt(model_type="2PL", n_latent=2, loadings=Confirmatory(Q, positive))
fit = fit_mcmc(model, responses)
idata = fit.to_idata(coords={"latent": ["positive_affect", "negative_affect"], "item": ITEMS})
```

Feel free to take a look at {ref}`other supported loading structures <api-loading-structures>`. In most cases you will likely want to stick with {py:func}`Confirmatory() <dynamirt.Confirmatory>`. Through the {py:func}`Sparsity() <dynamirt.Sparsity>` factory, dynamirt also includes experimental support for exploratory set ups by assuming a sparse loading structure with regularized horseshoe priors. This encourages a sparse orientation but does not guarantee identification. The results still suffer from sign and permutation invariances.

The loading structure itself will appear in the trace (e.g., in the `idata` object) under the name `"loadings"` `(n_items, n_latent)`. All related parameters will have the `"loadings."` prefix.

When you are sticking with the default parametrization of the latent construct as in all the examples above (e.g., no custom time series structure), you can also estimate the correlation between latents by setting `corr=True` in {py:func}`dynamirt() <dynamirt.dynamirt>`. The cholesky decomposition of the correlation matrix will show up as `"latent.residuals.correlation_cholesky"` `(n_latent, n_latent)` in the trace when `n_latent > 1`.

## Dichotomous Measurement Models

The above examples all make use of the two parameter logistic (2PL) model. All other binary measurement models work just the same and you can simply replace the `model_type` string in {py:func}`dynamirt() <dynamirt.dynamirt>` to your preferred measurement model. 

The 1PL model forces the loading structure to {py:func}`Fixed() <dynamirt.Fixed>` and will throw an error if you supply one manually. The 3PL and 4PL models can be weakly identified especially if you have little data.

## Polytomous Measurement Models

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

[Covariates & Built-In Terms](terms.md) · [Fitting Models](fitting.md) · [Static tutorial](../tutorials/tutorial_static.ipynb)
