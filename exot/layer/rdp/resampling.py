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
"""Resampling RDP layers"""
import copy
import math
import typing as t

import numpy as np
import pandas as pd
import scipy.signal
from scipy import interpolate

from exot.exceptions import *
from exot.util.attributedict import LabelMapping
from exot.util.misc import get_valid_access_paths, getitem, is_scalar_numeric
from exot.util.wrangle import *

from .._base import Layer


class Normalise(Layer, layer=Layer.Type.PrePost):
    def __init__(
        self,
        *,
        resampling_period_s: float,
        offset: float,
        environments_apps_zones: t.Mapping,
        **kwargs,
    ):
        """Initialise the Normalisation RDP layer

        Args:
            resampling_period (float): the resampling period
            offset (float): the offset of the measured value which will be compensated
            environments_apps_zones (t.Mapping): the env->app->zone mapping
        """
        self.resampling_period_s = resampling_period_s
        self.offset = offset
        self.environments_apps_zones = environments_apps_zones

    @property
    def _encode_types(self) -> t.Tuple[type]:
        return (np.ndarray,)

    @property
    def _decode_types(self) -> t.Tuple[type]:
        return (pd.DataFrame,)

    @property
    def _encode_validators(self):
        return {np.ndarray: [lambda v: v.ndim == 1]}

    @property
    def _encode_output_validators(self):
        return {
            pd.DataFrame: [
                lambda v: v.ndim == 2,
                lambda v: v.shape[1] == 1 + len(self.cores_and_schedules),
            ]
        }

    @property
    def _decode_validators(self):
        # Accepts: timestamps + multiple value
        return {pd.DataFrame: [lambda v: v.ndim >= 2, lambda v: v.shape[1] == 2]}

    @property
    def _decode_output_validators(self):
        # Accepts: timestamps + multiple value
        return {pd.DataFrame: [lambda v: v.ndim == 2, lambda v: v.shape[1] >= 2]}

    @property
    def requires_runtime_config(self) -> (bool, bool):
        """Does the layer's (encode, decode) require runtime configuration?"""
        return (False, True)

    @property
    def required_config_keys(self):
        """The required config keys

        Implements the `required_config_keys` from Configurable base class
        """
        return ["env", "mapping"]

    def validate(self) -> t.NoReturn:
        """Validate config

        Implementation of Configurable's `validate` method
        """
        try:
            if "env" in self.config:
                assert isinstance(self.config.env, str), ("env", str, type(self.config.env))
            if "mapping" in self.config:
                assert isinstance(self.config.mapping, tuple), (
                    "Mapping",
                    tuple,
                    type(self.config.mapping),
                )
                assert (
                    len(self.config.mapping) == 2
                ), "Mapping needs to have two elements (Matcher, dict)"
                assert isinstance(self.config.mapping[0], Matcher), (
                    "Mapping[0]",
                    Matcher,
                    type(self.config.mapping[0]),
                )
                assert isinstance(self.config.mapping[1], (LabelMapping, dict)), (
                    "Mapping[1]",
                    (LabelMapping, dict),
                    type(self.config.mapping[1]),
                )
        except AssertionError as e:
            raise MisconfiguredError("timevalue: {} expected {}, got: {}".format(*e.args[0]))

    @property
    def resampling_period_s(self):
        """Get the sampling period"""
        return self._resampling_period_s

    @resampling_period_s.setter
    def resampling_period_s(self, value):
        """Set the sampling period"""
        if not is_scalar_numeric(value):
            raise LayerMisconfigured("sampling_period must be an integer of float")
        self._resampling_period_s = value

    def _encode(self, lnestream: np.ndarray) -> pd.DataFrame:
        """
        For now this layer is only intended for use with AppExec Experiments.
        Therefore this method is only implmented as pass through.

        Args:
            lnestream (np.ndarray): the line-encoded stream from an LNE coder

        Returns:
            pd.DataFrame: the lnestream
        """
        return lnestream

    def _decode(self, rdpstream: pd.DataFrame) -> np.ndarray:
        """Resample and reshape an input rdpstream. A mapping is applied to non-numeric columns
        to convert them to numeric values.

        Args:
            rdpstream (pd.DataFrame): the rdpstream DataFrame produced by the I/O layer

        Returns:
            np.ndarray: a resampled and reshaped array, of width that is a multiple of
            the subsymbol_count, in the range [2 × subsymbol_count, 4 × subsymbol_count]
        """
        n_steps = math.floor(
            (rdpstream[rdpstream.columns[0]].iloc[-2] - rdpstream[rdpstream.columns[0]].iloc[0])
            / self.resampling_period_s
        )
        lnestream = np.empty((n_steps + 1, rdpstream.shape[1]))
        lnestream[:, 0] = np.linspace(0, n_steps * self.resampling_period_s, n_steps + 1)
        cols_to_map = rdpstream[self.config.mapping[0]].columns
        normalisation = self.environments_apps_zones[self.config.env]["snk"]["zone_config"][
            "T_norm"
        ]
        for idx in range(1, len(rdpstream.columns)):
            key = rdpstream.columns[idx]
            if key in cols_to_map:
                tmp = np.full(rdpstream[key].shape, self.config.mapping[1]["UNKNOWN"]["int"])
                for map_key in self.config.mapping[1]:
                    tmp[rdpstream[key] == map_key] = self.config.mapping[1][map_key]["int"]
                non_numeric_interpolation = interpolate.interp1d(
                    rdpstream[rdpstream.columns[0]], tmp, kind="nearest"
                )
                lnestream[:, idx] = non_numeric_interpolation(lnestream[:, 0])
            elif np.issubdtype(rdpstream[key], np.number):
                if key in normalisation.keys():
                    lnestream[:, idx] = np.interp(
                        lnestream[:, 0],
                        rdpstream[rdpstream.columns[0]],
                        rdpstream[key] / normalisation[key] - self.offset,
                    )
                else:
                    lnestream[:, idx] = np.interp(
                        lnestream[:, 0],
                        rdpstream[rdpstream.columns[0]],
                        rdpstream[key] - self.offset,
                    )
            else:
                raise Exception(f"No mapping found for column {key}")

        return lnestream
