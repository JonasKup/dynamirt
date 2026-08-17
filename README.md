# dynamirt

Whenever we measure sets of items for multiple respondents, a common assumption is that the measured items arise from a limited number of latent traits. When we measure the same set of items for the same respondent repeatedly (panel studies) the response data often arise in a ragged, inhomogenous fashion, meaning respondents don't share the same number of time points and are sampled continuously rather than at fixed intervals. This is the case, for example, in ecological momentary assessment studies like e-diaries or diagnosis codes in electronic health records. For these cases, the dynamirt package helps to recover continuous latent trajectories under the longitudinal item response theory framework using exact and Hilbert space approximate Gaussian processes.

<table>
  <tr>
    <td><img src="imgs/simulated_item_responses.svg" width="400" alt="Item Response Pattern"></td>
    <td><img src="imgs/recovered_trajectories.svg" width="400" alt="Latent Trajectories"></td>
  </tr>
</table>

## Quickstart

### Static 2-PL MIRT model

```python
from dynamirt.api import dynamirt
from dynamirt.fit import fit_mcmc
from dynamirt.loadings import Confirmatory

model = dynamirt(
    model_type="2PL",
    n_latent=2,
    loadings=Confirmatory(Q, positive_anchors=Q),
)

covariates = {}
idata, mcmc = fit_mcmc(model, responses, covariates)
```

### Longitudinal 2-PL MIRT model with HSGPs

```python
from dynamirt.api import dynamirt
from dynamirt.fit import fit_svi
from dynamirt.loadings import Sparsity
from dynamirt.terms import Linear, HSGP

covariates = {
  "respondent_id": ...,
  "time": ...} 

intercept = Linear("respondent_mean", "one_", group_by="respondent_id")
trajectory = HSGP(
  name="trajectory", 
  predictors="time", 
  group_by="respondent_id", 
  kernel="Matern",
  ell= 1.5, # assumes time is scaled and centered
  m=30,     # number of basis functions
  nu=3/2    # smoothness of the Matérn kernel
)

model = dynamirt(
    model_type="2PL",
    n_latent=2,
    loadings=Sparsity(),
    latent_fn=[intercept, trajectory]
)

idata, guide, svi_result = fit_svi(model, responses, covariates)
```

Links to docs etc

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