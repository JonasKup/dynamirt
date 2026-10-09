# Custom Terms

While the built-ins lean toward time series modeling, they don't exhaustively cover all longitudinal models you may want to build and maybe you want to build something else entirely (e.g., a spatial model). 

Creating custom terms is relatively straight forward because you just need to write a NumPyro/JAX function that implements your desired operation. The most important requirement for custom term functions is that they return their result as an array of shape `(n_obs, n_target)`. `n_target` should be the latent dimension if used as a `latent_term` or the number of items if used as an `item_term`. 

All custom term functions must take two arguments: a dynamirt {py:class}`ModelContext <dynamirt.ModelContext>` and `n_target`. The ModelContext lets you access the `responses` array, the `covariates` dict, as well as `n_obs`, `n_var` (number of items), `n_latent`, and `index_sizes`.

As an example let's create a GPMHT style model ([McBride & Wang, 2026](https://doi.org/10.1017/psy.2026.10082)) in which respondents share a B-spline mean function over time and HSGPs describe respondent specific deviations. dynamirt does not have built-in support for splines so we create a custom term. We precompute the spline basis and supply it to the `spline_mean` function through the `covariates` dict. Finally, we wrap it in {py:class}`CustomTerm() <dynamirt.CustomTerm>` and hand it to the model constructor together with the built-in HSGPs:

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

For an additional example where the latent space is parametrized by a neural differential equation refer to [this notebook](../tutorials/tutorial_NODE.ipynb).

[Covariates & Built-In Terms](terms.md) · [Priors, Parameters & Multilevel Models](parameters.md)
