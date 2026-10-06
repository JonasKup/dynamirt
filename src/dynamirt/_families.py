from .gllvm.context import ModelContext
from .gllvm.families import Bernoulli

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

import numpyro
import numpyro.distributions as dist

from typing import Literal

def _dichotomous(
    model_type: Literal["1PL", "2PL", "3PL", "4PL"],
    lower_asymptote_prior: dist.Distribution = None,
    upper_asymptote_gap_prior: dist.Distribution = None
    ):
    
    """Compute likelihood for 1PL, 2PL, 3PL, 4PL models"""
    
    lower_asymptote_prior = dist.Beta(2, 8) if lower_asymptote_prior is None else lower_asymptote_prior
    upper_asymptote_gap_prior = dist.Beta(8, 2) if upper_asymptote_gap_prior is None else upper_asymptote_gap_prior

    if model_type in ["1PL", "2PL"]:
        return Bernoulli()

    def family(eta: ArrayLike, ctx: ModelContext):
        

        la = numpyro.sample("lower_asymptote", lower_asymptote_prior.expand((ctx.n_var,)).to_event(1))

        if model_type == "4PL":
            # sample gap between ua and la so la > ua can never happen
            gap = numpyro.sample("asymptote_gap", upper_asymptote_gap_prior.expand((ctx.n_var,)).to_event(1))
            ua = numpyro.deterministic("upper_asymptote", la + (1.0 - la) * gap)
        else:
            ua = 1.0

        p = la + (ua - la) * jax.nn.sigmoid(eta)
        p = jnp.clip(p, 1e-7, 1.0 - 1e-7) # guard the log-likelihood
        
        return dist.BernoulliProbs(probs=p)

    family.n_cat = 2
    return family

def _grm_cutpoints(probabilities):
    # logit(cumulative probability), using tail sums to avoid rounding to one.
    below = jnp.cumsum(probabilities, axis=-1)[..., :-1]
    above = jnp.cumsum(probabilities[..., ::-1], axis=-1)[..., ::-1][..., 1:]
    return jnp.log(below) - jnp.log(above)


def _polytomous(
    model_type, 
    n_cat, 
    prior=None, 
    alpha=1.0):
    
    """Compute likelihood for GRM, PCM, GPCM models with a shared or per-item category count.

    GRM baseline probabilities at predictor zero have a Dirichlet(alpha) prior.
    alpha is a positive scalar or a vector of length max(n_cat); shorter items
    use its first n_cat[j] entries. Cutpoints are logits of cumulative probabilities.
    prior controls PCM/GPCM steps. Per-item cutpoints/steps are zero-padded;
    only the first n_cat[j] - 1 entries of item j are valid.
    """
    
    prior = dist.Normal(0, 1) if prior is None else prior

    counts = np.asarray(n_cat)
    
    if (counts.ndim > 1 or counts.size == 0
            or counts.dtype.kind not in "iu" or np.any(counts < 2)):
        raise ValueError("n_cat must contain integers >= 2")
    
    if counts.ndim == 0:
        n_cat = int(counts)
        
    if model_type == "GRM":
        concentration = np.asarray(alpha)
        if (concentration.ndim > 1
                or (concentration.ndim == 1 and concentration.shape != (int(counts.max()),))
                or not np.all(np.isfinite(concentration)) or np.any(concentration <= 0)):
            raise ValueError("alpha must be a positive scalar or a vector of length max(n_cat)")
        concentration = jnp.broadcast_to(jnp.asarray(concentration, dtype=float), (int(counts.max()),))
        
    if counts.ndim:
        max_cat = int(counts.max())
        valid = np.arange(max_cat - 1) < counts[:, None] - 1
        item, threshold = np.nonzero(valid)
        categories = jnp.arange(max_cat)

    def family(eta, ctx):
        if counts.ndim and counts.shape != (ctx.n_var,):
            raise ValueError("n_cat must have one count per item")
        if counts.ndim:
            if model_type == "GRM":
                c = jnp.zeros(valid.shape)
                for count in np.unique(counts):
                    rows = np.flatnonzero(counts == count)
                    p = numpyro.sample(
                        f"baseline_category_probs_{count}",
                        dist.Dirichlet(concentration[:count]).expand((len(rows),)).to_event(1),
                    )
                    c = c.at[rows, :count - 1].set(_grm_cutpoints(p))
                numpyro.deterministic("cutpoints", c)
                last = c[jnp.arange(ctx.n_var), counts - 2]
                # OrderedLogistic needs ordered filler thresholds, even though
                # their categories are subsequently masked out.
                c = jnp.where(valid, c, last[:, None] + jnp.arange(max_cat - 1))
                logits = dist.OrderedLogistic(eta, c).logits
                # The final valid category includes the entire upper tail.
                logits = jnp.where(
                    categories == counts[:, None] - 1,
                    jax.nn.log_sigmoid(eta - last)[..., None],
                    logits,
                )
            else:
                raw = numpyro.sample("steps_raw", prior.expand((len(item),)).to_event(1))
                d = numpyro.deterministic(
                    "steps", jnp.zeros(valid.shape).at[item, threshold].set(raw)
                )
                logits = jnp.cumsum(
                    jnp.pad(eta[..., None] - d, ((0, 0), (0, 0), (1, 0))), axis=-1
                )
            # Structural zeros are valid probabilities, but -inf is not a
            # valid CategoricalLogits parameter under argument validation.
            # fix is to hand probs instead of logits to dist.Categorical
            masked_logits = jnp.where(categories < counts[:, None], logits, -jnp.inf)
            return dist.Categorical(probs=jax.nn.softmax(masked_logits, axis=-1))

        if model_type == "GRM":
            p = numpyro.sample(
                "baseline_category_probs",
                dist.Dirichlet(concentration).expand((ctx.n_var,)).to_event(1),
            )
            c = numpyro.deterministic("cutpoints", _grm_cutpoints(p))
            return dist.OrderedLogistic(eta, c)
        
        base = prior.expand((ctx.n_var, n_cat - 1)).to_event(1)
        d = numpyro.sample("steps", base.to_event(1))
        logits = jnp.cumsum(jnp.pad(eta[..., None] - d, ((0, 0), (0, 0), (1, 0))), axis=-1)
        return dist.CategoricalLogits(logits)

    family.n_cat = counts
    return family
