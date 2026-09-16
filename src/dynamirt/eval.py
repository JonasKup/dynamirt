"""Posterior trajectory and loading plots."""

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

def plot_loadings(
    idata: xr.DataTree,
    *,
    var_name: str = "loadings",
    item: str | int | Sequence[str | int] | None = None,
    latent: str | int | Sequence[str | int] | None = None,
    prob: float = 0.95,
    annotate: bool = True,
    cmap: str = "RdBu_r",
    vmax: float | None = None,
    colorbar: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Plot posterior median loadings as a heatmap.

    Args:
        idata: ArviZ data with an item-by-latent posterior variable.
        var_name: Posterior variable to plot. Defaults to "loadings".
        item: Item coordinate label or labels. None selects all. Defaults to None.
        latent: Latent coordinate label or labels. None selects all. Defaults to None.
        prob: Probability mass of annotated HDIs. Defaults to 0.95.
        annotate: Show medians and HDIs; bold means the HDI excludes zero.
            Defaults to True.
        cmap: Matplotlib colormap. Defaults to "RdBu_r".
        vmax: Positive limit for the symmetric color range. None uses the
            largest absolute median, or 1 for all-zero medians. Defaults to None.
        colorbar: Add a colorbar to the figure. Defaults to True.
        ax: Axes to draw on. Creates one if None. Defaults to None.

    Returns:
        The matplotlib Axes. Titles and figure layout are left to the caller.
    """
    values = idata["posterior"][var_name]
    for dim, selection in (("item", item), ("latent", latent)):
        if selection is not None:
            labels = [selection] if np.isscalar(selection) else list(selection)
            values = values.sel({dim: labels})
    sample_dims = [dim for dim in ("chain", "draw", "sample") if dim in values.dims]
    median = values.median(dim=sample_dims).transpose("item", "latent").values
    if annotate:
        interval = az.hdi(values, prob=prob, dim=sample_dims)
        lo = interval.sel(ci_bound="lower").transpose("item", "latent").values
        hi = interval.sel(ci_bound="upper").transpose("item", "latent").values

    if vmax is None:
        vmax = float(np.abs(median).max()) or 1.0
    if vmax <= 0:
        raise ValueError("vmax must be positive")
    n_items, n_lat = median.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(1.7 * n_lat + 2, 0.75 * n_items + 1.5))
    im = ax.imshow(median, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    if annotate:
        for i in range(n_items):
            for j in range(n_lat):
                # Choose black or white by contrast against the displayed cell.
                rgba = np.asarray(im.cmap(im.norm(median[i, j])))
                rgb = rgba[:3] * rgba[3] + np.asarray(ax.get_facecolor()[:3]) * (1 - rgba[3])
                linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
                ax.text(
                    j, i, f"{median[i, j]:.2f}\n[{lo[i, j]:.2f}, {hi[i, j]:.2f}]",
                    ha="center", va="center", fontsize=8,
                    color="black" if luminance > 0.179 else "white",
                    fontweight="bold" if (lo[i, j] > 0) or (hi[i, j] < 0) else "normal",
                )

    ax.set_xticks(range(n_lat), values["latent"].values)
    ax.set_yticks(range(n_items), values["item"].values)
    ax.set_xticks(np.arange(n_lat + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_items + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", lw=1.5)
    ax.tick_params(which="minor", length=0)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, shrink=0.8)
    return ax
