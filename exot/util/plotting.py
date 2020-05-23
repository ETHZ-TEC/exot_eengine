"""Plotting support"""

import contextlib
import inspect
import pathlib
import typing as t

import matplotlib.pyplot as plt
import matplotlib.transforms as tx
import numpy as np
import pandas
import seaborn as sns
from matplotlib.collections import LineCollection

from exot.experiment.frequency_sweep import FrequencySweepRun
from exot.experiment.performance import PerformanceRun
from exot.util.attributedict import AttributeDict
from exot.util.scinum import is_fitted, unpack_array

__all__ = ("_save_path_helper", "remove_spine", "add_spine", "rugplot")


def _save_path_helper(path: t.Union[pathlib.Path, str]) -> pathlib.Path:
    # Check and normalise variable type
    if not isinstance(path, (str, pathlib.Path)):
        raise TypeError(f"wrong type supplied for save directory path", type(path))
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    if path.exists() and not path.is_dir():
        raise ValueError("provided a file instead of a directory", path)

    # Directory creation can fail, raising, for example, a PermissionError.
    if not path.exists():
        path.mkdir(parents=True)

    if not path.exists() and path.is_dir():
        raise RuntimeError("postcondition failed: directory is created and available")

    return path


def remove_spine(axis, which, ticks_only=False):
    if not ticks_only:
        axis.spines[which].set_color("none")
    if which in ["top", "bottom"]:
        axis.tick_params(axis="x", color=(0, 0, 0, 0))
    else:
        axis.tick_params(axis="y", color=(0, 0, 0, 0))


def add_spine(axis, which, ticks_only=False):
    if not ticks_only:
        axis.spines[which].set_color((0, 0, 0, 1))
    params = {which: True}
    if which in ["top", "bottom"]:
        axis.tick_params(axis="x", color=(0, 0, 0, 1), **params)
    else:
        axis.tick_params(axis="y", color=(0, 0, 0, 1), **params)


def rugplot(a, height=0.05, axis="x", ax=None, top=False, **kwargs):
    """Plot datapoints in an array as sticks on an axis. Adapted from seaborn.

    Args:
        a (vector): 1D array of observations.
        height (scalar, optional): Height of ticks as proportion of the axis.
        axis ({'x' | 'y'}, optional): Axis to draw rugplot on.
        ax (matplotlib axes, optional): Axes to draw plot into; otherwise grabs current axes.
        **kwargs: Other keyword arguments are passed to ``LineCollection``

    Returns:
        ax (matplotlib axes): The Axes object with the plot on it.
    """
    if ax is None:
        ax = plt.gca()
    a = np.asarray(a)
    vertical = kwargs.pop("vertical", axis == "y")

    alias_map = dict(linewidth="lw", linestyle="ls", color="c")
    for attr, alias in alias_map.items():
        if alias in kwargs:
            kwargs[attr] = kwargs.pop(alias)
    kwargs.setdefault("linewidth", 1)

    line = [0, height] if not top else [1, 1 - height]

    if vertical:
        trans = tx.blended_transform_factory(ax.transAxes, ax.transData)
        xy_pairs = np.column_stack([np.tile(line, len(a)), np.repeat(a, 2)])
    else:
        trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
        xy_pairs = np.column_stack([np.repeat(a, 2), np.tile(line, len(a))])
    line_segs = xy_pairs.reshape([len(a), 2, 2])
    ax.add_collection(LineCollection(line_segs, transform=trans, **kwargs))

    ax.autoscale_view(scalex=not vertical, scaley=vertical)

    return ax
