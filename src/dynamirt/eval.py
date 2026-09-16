"""Posterior trajectory plotting."""

from collections.abc import Mapping, Sequence

import arviz as az
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import xarray as xr


def plot_trajectories(
    idata: xr.DataTree,
    *,
    time: str,
    coords: Mapping[str, object] | None = None,
    var_name: str = "theta",
    latent: str | int | Sequence[str | int] | None = None,
    prob: float = 0.95,
    ax: Axes | None = None,
) -> Axes:
    """Plot one respondent's latent medians and pointwise posterior HDIs.

    Args:
        idata: ArviZ data with an obs-by-latent posterior variable.
        time: Name of the observation coordinate containing time values.
        coords: Observation coordinate selections, e.g. {"respondent_id": 42}.
            Each value must be scalar; selections are combined. With None,
            the data must already describe one respondent. Defaults to None.
        var_name: Posterior trajectory variable. Defaults to "theta".
        latent: Latent coordinate label or labels, including integer labels.
            None plots all latents. Defaults to None.
        prob: Probability mass of the pointwise HDI. Defaults to 0.9.
        ax: Axes to draw on. Creates one if None. Defaults to None.

    Returns:
        The matplotlib Axes. Lines are labeled using latent coordinates;
        call ax.legend() to display a legend.

    Raises:
        KeyError: If a requested variable or coordinate label is missing.
        ValueError: If dimensions or selections are invalid, no observations
            remain, or selected times are missing or duplicated.
    """
    values = idata["posterior"][var_name]
    sample_dims = [dim for dim in ("chain", "draw", "sample") if dim in values.dims]
    if (
        set(sample_dims) not in ({"chain", "draw"}, {"draw"}, {"sample"})
        or set(values.dims) != {*sample_dims, "obs", "latent"}
    ):
        raise ValueError(
            f"{var_name!r} must have obs and latent dimensions plus chain/draw or sample"
        )

    for name, label in (coords or {}).items():
        # if name not in values.coords:
        #     raise KeyError(f"Observation coordinate {name!r} is missing")
        # if values[name].dims != ("obs",) or np.ndim(label) != 0:
        #     raise ValueError("coords must select scalar values of observation coordinates")
        values = values.isel(obs=values[name] == label)
    # if not values.sizes["obs"]:
    #     raise ValueError("No observations match coords")
    # if time not in values.coords:
    #     raise KeyError(f"Time coordinate {time!r} is missing")
    # if values[time].dims != ("obs",):
    #     raise ValueError("The time coordinate must have dimension obs")
    # if np.unique(values[time].values).size != values.sizes["obs"]:
    #     raise ValueError("Selected times must be unique; select one respondent/session or aggregate first")
    
    values = values.sortby(time)
    if latent is not None:
        labels = [latent] if np.isscalar(latent) else list(latent)
        values = values.sel(latent=labels)
    # if not values.sizes["latent"]:
    #     raise ValueError("Select at least one latent")

    median = values.median(dim=sample_dims)
    interval = az.hdi(values, dim=sample_dims, prob=prob)
    if ax is None:
        _, ax = plt.subplots()
    for label in values.latent.values:
        line, = ax.plot(
            values[time].values, median.sel(latent=label).values,
            marker="o", markersize=3, label=str(label),
        )
        ax.fill_between(
            values[time].values,
            interval.sel(latent=label, ci_bound="lower").values,
            interval.sel(latent=label, ci_bound="upper").values,
            color=line.get_color(), alpha=0.3,
        )
    ax.set_xlabel(time)
    ax.set_ylabel(var_name)
    return ax
