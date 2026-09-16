# dynamirt 🧨

API for Bayesian item response modeling in particular and generalized linear latent variable models more generally. Additionally, the latent space can be parametrized in a composable fashion 
through built-ins like HSGPs as well as custom extensions that should make it an attractive choice for latent variable modeling in settings with correlated measurements such as longitudinal studies.

<table>
  <tr>
    <td><img src="imgs/simulated_item_responses.svg" width="400" alt="Item Response Pattern"></td>
    <td><img src="imgs/recovered_trajectories.svg" width="400" alt="Latent Trajectories"></td>
  </tr>
</table>

## Installation

```bash
python -m pip install dynamirt
```

`dynamirt` requires Python 3.10 or newer.

## Quickstart

### Static 2-PL MIRT model

```python
from dynamirt import Confirmatory, dynamirt, fit_mcmc

model = dynamirt(
    model_type="2PL",
    n_latent=2,
    loadings=Confirmatory(Q, positive=Q),
)

covariates = {}
idata, mcmc = fit_mcmc(model, responses, covariates)
```

### Longitudinal 2-PL MIRT model with HSGPs

```python
from dynamirt import HSGP, Linear, Sparsity, dynamirt, fit_svi

covariates = {
    "respondent_id": ...,
    "time": ...,
}

intercept = Linear("respondent_mean", group_by="respondent_id")

trajectory = HSGP(
    name="trajectory",
    predictors="time",
    group_by="respondent_id",
    kernel="Matern",
    ell=1.5,  # assumes time is scaled and centered
    m=30,  # number of basis functions
    nu=3 / 2,  # smoothness of the Matern kernel
)

model = dynamirt(
    model_type="2PL",
    n_latent=2,
    loadings=Sparsity(),
    latent_fn=[intercept, trajectory],
)

idata, guide, svi_result = fit_svi(model, responses, covariates)
```

The API documentation is under development.

## Built with

- [NumPyro](https://num.pyro.ai/)
- [JAX](https://docs.jax.dev/en/latest/index.html)
- [ArviZ](https://www.arviz.org/en/latest/)
- [tinygp](https://tinygp.readthedocs.io/en/stable/)

## Other IRT and latent variable software

### R

- [mirt](https://github.com/philchalmers/mirt) — Multidimensional item response theory in R
- [emIRT](https://github.com/kosukeimai/emIRT) — EM Algorithms for Estimating Item Response Theory Models (including ideal point estimation over time)
- [gllvm](https://github.com/JenniNiku/gllvm) — Generalized linear latent variable models
- [Hmsc](https://github.com/hmsc-r/HMSC) — Hierarchical Modelling of Species Communities

### Python

- [py-irt](https://github.com/nd-ball/py-irt) — A Scalable Item Response Theory Library for Python
- [girth](https://github.com/eribean/girth) — A python package for estimating item response theory (IRT) parameters
