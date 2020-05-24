# Copyright (c) 2015-2020, Swiss Federal Institute of Technology (ETH Zurich)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
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
    """A helper function for save paths
    
    Args:
        path (t.Union[pathlib.Path, str]): The save path
    
    Raises:
        TypeError: Wrong type supplied
        ValueError: Provided a file instead of a directory
        RuntimeError: Directory was not created
    
    Returns:
        pathlib.Path: The save path
    """
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


def remove_spine(axis, which: str, ticks_only: bool = False) -> None:
    """Removes the spine from an axis
    
    Args:
        axis: The matplotlib axis
        which (str): Which spine to remove? (top, bottom)
        ticks_only (bool, optional): Remove only the ticks?. Defaults to False.
    """
    if not ticks_only:
        axis.spines[which].set_color("none")
    if which in ["top", "bottom"]:
        axis.tick_params(axis="x", color=(0, 0, 0, 0))
    else:
        axis.tick_params(axis="y", color=(0, 0, 0, 0))


def add_spine(axis, which: str, ticks_only: bool = False):
    """Adds a spine to an axis
    
    Args:
        axis: The matplotlib axis
        which (str): Which spine to add? (top, bottom)
        ticks_only (bool, optional): Add only the ticks?. Defaults to False.
    """
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
