# dynamirt

![Item Response Pattern](imgs/simulated_item_responses.svg)

![Posterior Recovery](imgs/recovered_trajectories.svg)

<img src="imgs/simulated_item_responses.svg" width="250" alt="Item Response Pattern">

## Quickstart

Fit a multidimensional longitudinal 2-PL IRT model with three latent trajectories and IID Matérn-Kernel HSGP priors on the trajectories.

```
from dynamirt.models import dynamirt
from dynamirt.fit import fit_mcmc
from dynamirt.dynamics import HSGP


model_hsgp = dynamirt(
    ir_model="2PL",
    model_type="confirmatory",
    n_latents=3,
    latent_dynamics_fn=HSGP,
    latent_dynamics_kwargs={ 
        "nu": 5/2,
        "ell": 1.5 * n_time,
        "m": n_time,
    }
)

idata, mcmc = fit_mcmc(
    model_hsgp, 
    response_data,
    anchors,
    seed=0)

```

Plot response data next to estimated latents for person 0:

```
from dynamirt.eval import plot_summary_by_latent

plot_summary_by_latent(idata, 0)
```