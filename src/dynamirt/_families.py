from .gllvm.context import ModelContext

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
    upper_asymptote_gap_prior: dist.Distribution = None,
    intercept_prior: dist.Distribution = None,
    ):
    
    """Compute likelihood for 1PL, 2PL, 3PL, 4PL models"""
    
    lower_asymptote_prior = dist.Beta(2, 8) if lower_asymptote_prior is None else lower_asymptote_prior
    upper_asymptote_gap_prior = dist.Beta(8, 2) if upper_asymptote_gap_prior is None else upper_asymptote_gap_prior

    intercept_prior = dist.Normal(0, 2) if intercept_prior is None else intercept_prior

    def family(eta: ArrayLike, ctx: ModelContext):
        intercept = numpyro.sample("intercept", intercept_prior.expand((ctx.n_var,)).to_event(1))
        eta = eta + intercept
        if model_type in ["1PL", "2PL"]:
            return dist.Bernoulli(logits=eta)

        la = numpyro.sample("lower_asymptote", lower_asymptote_prior.expand((ctx.n_var,)).to_event(1))

        if model_type == "4PL":
            # sample a fraction of the remaining interval so ua stays above la
            gap = numpyro.sample("upper_asymptote_fraction", upper_asymptote_gap_prior.expand((ctx.n_var,)).to_event(1))
            ua = numpyro.deterministic("upper_asymptote", la + (1.0 - la) * gap)
        else:
            ua = 1.0

        p = la + (ua - la) * jax.nn.sigmoid(eta)
        p = jnp.clip(p, 1e-7, 1.0 - 1e-7) # guard the log-likelihood
        
        return dist.BernoulliProbs(probs=p)

    family.n_cat = 2
    return family

def _category_layout(n_cat):
    """Validate category counts and prepare fixed indices for ordinal families."""
    counts = np.asarray(n_cat)
    if (counts.ndim > 1 or counts.size == 0
            or counts.dtype.kind not in "iu" or np.any(counts < 2)):
        raise ValueError("n_cat must contain integers >= 2")
    max_cat = int(counts.max())
    if not counts.ndim:
        return counts, max_cat, None, None, None
    valid = np.arange(max_cat - 1) < counts[:, None] - 1
    item, threshold = np.nonzero(valid)
    return counts, max_cat, valid, item, threshold


def _cutpoints_from_probs(probabilities):
    """Logits of cumulative probabilities, using tail sums to avoid rounding to one."""
    below = jnp.cumsum(probabilities, axis=-1)[..., :-1]
    above = jnp.cumsum(probabilities[..., ::-1], axis=-1)[..., ::-1][..., 1:]
    return jnp.log(below) - jnp.log(above)


def _masked_categorical(logits, counts):
    """Keep padded categories at exactly zero probability."""
    categories = jnp.arange(logits.shape[-1])
    masked_logits = jnp.where(categories < counts[:, None], logits, -jnp.inf)
    return dist.Categorical(probs=jax.nn.softmax(masked_logits, axis=-1))


def _grm(n_cat, *, alpha=1.0):
    """GRM with Dirichlet baseline probabilities and derived ordered cutpoints.

    alpha is a positive scalar, or a category vector when all counts are equal.
    Per-item cutpoints are zero-padded.
    """
    counts, max_cat, valid, _, _ = _category_layout(n_cat)

    concentration = np.asarray(alpha)
    if (concentration.ndim > 1
            or (concentration.ndim == 1 and (
                concentration.shape != (max_cat,)
                or np.any(counts != max_cat)))
            or not np.all(np.isfinite(concentration)) or np.any(concentration <= 0)):
        raise ValueError("alpha must be finite and positive; a category vector requires equal n_cat and length n_cat")
    concentration = jnp.broadcast_to(jnp.asarray(concentration, dtype=float), (max_cat,))
    if counts.ndim:
        groups = tuple((int(count), np.flatnonzero(counts == count))
                       for count in np.unique(counts))

    def family(eta, ctx):
        if counts.ndim and counts.shape != (ctx.n_var,):
            raise ValueError("n_cat must have one count per item")
        if counts.ndim:
            c = jnp.zeros(valid.shape)
            for count, rows in groups:
                p = numpyro.sample(
                    f"baseline_probs_{count}",
                    dist.Dirichlet(concentration[:count]).expand((len(rows),)).to_event(1),
                )
                c = c.at[rows, :count - 1].set(_cutpoints_from_probs(p))
            numpyro.deterministic("cutpoints", c)
            last = c[jnp.arange(ctx.n_var), counts - 2]
            # OrderedLogistic needs ordered filler thresholds, even though
            # their categories are subsequently masked out.
            c = jnp.where(valid, c, last[:, None] + jnp.arange(max_cat - 1))
            logits = dist.OrderedLogistic(eta, c).logits
            # The final valid category includes the entire upper tail.
            logits = jnp.where(
                jnp.arange(max_cat) == counts[:, None] - 1,
                jax.nn.log_sigmoid(eta - last)[..., None],
                logits,
            )
            return _masked_categorical(logits, counts)

        p = numpyro.sample(
            "baseline_probs",
            dist.Dirichlet(concentration).expand((ctx.n_var,)).to_event(1),
        )
        c = numpyro.deterministic("cutpoints", _cutpoints_from_probs(p))
        return dist.OrderedLogistic(eta, c)

    family.n_cat = counts
    return family


def _partial_credit(n_cat, *, step_prior=None):
    """PCM/GPCM likelihood; the model builder selects fixed/free loadings.

    step_prior defaults to Normal(0, 2). Per-item steps are zero-padded;
    only the first n_cat[j] - 1 entries of item j are valid.
    """
    step_prior = dist.Normal(0, 2) if step_prior is None else step_prior
    counts, max_cat, valid, item, threshold = _category_layout(n_cat)

    def family(eta, ctx):
        if counts.ndim and counts.shape != (ctx.n_var,):
            raise ValueError("n_cat must have one count per item")
        if counts.ndim:
            raw = numpyro.sample("steps.raw", step_prior.expand((len(item),)).to_event(1))
            d = numpyro.deterministic(
                "steps", jnp.zeros(valid.shape).at[item, threshold].set(raw)
            )
            logits = jnp.cumsum(
                jnp.pad(eta[..., None] - d, ((0, 0), (0, 0), (1, 0))), axis=-1
            )
            return _masked_categorical(logits, counts)

        base = step_prior.expand((ctx.n_var, max_cat - 1)).to_event(1)
        d = numpyro.sample("steps", base.to_event(1))
        logits = jnp.cumsum(jnp.pad(eta[..., None] - d, ((0, 0), (0, 0), (1, 0))), axis=-1)
        return dist.CategoricalLogits(logits)

    family.n_cat = counts
    return family
