# Fitting Models

Models can be fit either through {py:func}`fit_mcmc(model, responses, covariates, ...) <dynamirt.fit_mcmc>` for Markov chain Monte Carlo or {py:func}`fit_svi(model, responses, covariates, ...) <dynamirt.fit_svi>` for stochastic variational inference. Both return a {py:class}`~dynamirt.fit.FitResult` object that contains the original NumPyro MCMC and SVI machinery in `result.inference`. It also provides a unified {py:meth}`.to_idata <dynamirt.fit.FitResult.to_idata>` function for both SVI and MCMC inference that creates ArviZ compatible inference data.

The output of the {py:func}`dynamirt() <dynamirt.dynamirt>` model constructor is simply a NumPyro model so you don't need to make use of the above helpers if you prefer to stick directly with NumPyro for inference. The only drawback to that is that the built-in helpers set some dimension names that subsequent dynamirt plotting functions make use of.

[Building Models](models.md) · [Plots](plots.md) · [Static tutorial](../tutorials/tutorial_static.ipynb) · [Dynamic tutorial](../tutorials/tutorial_dynamic.ipynb)
