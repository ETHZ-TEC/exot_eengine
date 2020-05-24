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
"""Quantising RDP layer as used for the power-cc"""

import copy
import typing as t

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.signal

from exot.exceptions import *
from exot.util.misc import (
    get_cores_and_schedules,
    get_valid_access_paths,
    getitem,
    is_scalar_numeric,
)

from .._base import Layer
from .._mixins import RDPmixins
from .coreactivation import CoreActivation


class FrequencyLevelQuantistion(RDPmixins, Layer, layer=Layer.Type.PrePost):
    def __init__(self, *, timeout_s: int = 10, environments_apps_zones: t.Mapping, **kwargs):
        """Initialise the Conservative Governor Line Coding layer

        Args:
        """
        self.timeout_s = timeout_s
        super().__init__(**kwargs)
        self.cores_and_schedules = get_cores_and_schedules(environments_apps_zones)

    @property
    def required_config_keys(self):
        """The required config keys

        Implements the `required_config_keys` from Configurable base class
        """
        return ["env"]

    def _encode(self, lnestream):
        tag_count = len(self.cores_and_schedules)
        rdpstream = np.empty((lnestream.shape[0], tag_count), dtype=np.dtype("int"))
        tags = []
        for idx, (core_count, tag) in enumerate(self.cores_and_schedules):
            tags.append(tag)
            rdpstream[:, idx] = lnestream

        return pd.DataFrame.join(
            pd.DataFrame(np.full(lnestream.shape, self.timeout_s), columns=["timestamp"]),
            pd.DataFrame(rdpstream, columns=tags),
        )

    def _decode(self, rdpstream: pd.DataFrame) -> np.ndarray:
        thresholds = self.config.environments_apps_zones[self.config.env]["snk"][
            "zone_config"
        ].frequency_thresholds
        lnestream = cp.deepcopy(rdpstream.iloc[:, 1].to_numpy())
        for tidx in range(len(thresholds)):
            if tidx < len(thresholds) - 1:
                lnestream[
                    np.logical_and(
                        lnestream >= thresholds[tidx], lnestream < thresholds[tidx + 1]
                    )
                ] = tidx
            else:
                lnestream[lnestream >= thresholds[tidx]] = tidx

        return lnestream


"""
QuantCoreActivation
--------------
Quantising RDP layer as used for the power-cc

"""


class QuantCoreActivation(CoreActivation):
    @property
    def required_config_keys(self):
        """The required config keys

        Implements the `required_config_keys` from Configurable base class
        """
        return ["symbol_rate", "subsymbol_rate", "rdpstream", "env"]

    def _decode(self, rdpstream: pd.DataFrame) -> np.ndarray:
        """Resample and reshape an input rdpstream

        Args:
            rdpstream (pd.DataFrame): the rdpstream DataFrame produced by the I/O layer

        Returns:
            np.ndarray: a resampled and reshaped array, of width that is a multiple of
            the subsymbol_count, in the range [2 × subsymbol_count, 4 × subsymbol_count]
        """
        timestamps = rdpstream.iloc[:, 0]

        actual_start = timestamps.iloc[0]
        actual_end = timestamps.iloc[len(timestamps) - 1]

        orig_samples_per_symbol: float
        sampling_period_inferred = (
            0.15 * timestamps.diff().mean() + 0.85 * timestamps.diff().median()
        )

        if abs(sampling_period_inferred - self.sampling_period) / self.sampling_period > 0.1:
            pass

        orig_samples_per_symbol = 1 / (self.sampling_period * self.config.symbol_rate)
        subsymbol_count = self.config.subsymbol_rate / self.config.symbol_rate
        self._new_samples_per_symbol = max([subsymbol_count * 100, orig_samples_per_symbol])

        # make sure that the _samples_per_symbol is a multiple of subsymbol_count
        if self._new_samples_per_symbol % subsymbol_count != 0:
            self._new_samples_per_symbol = subsymbol_count * np.ceil(
                self._new_samples_per_symbol / subsymbol_count
            )

        # make sure that _samples_per_symbol is an integer
        if not self._new_samples_per_symbol.is_integer():
            self._new_samples_per_symbol = np.ceil(self._new_samples_per_symbol)
        else:
            self._new_samples_per_symbol = int(self._new_samples_per_symbol)

        self._resampling_factor = self._new_samples_per_symbol / orig_samples_per_symbol

        # Median filter
        window = min([max([2 * round(orig_samples_per_symbol / 3) + 1, 1]), 9])
        values = scipy.signal.medfilt(rdpstream.iloc[:, 1].to_numpy(), kernel_size=window)

        # set-up resampling
        original_size = len(timestamps)
        original_indexer = timestamps.to_numpy()

        self._oversampling_period = self.sampling_period / self._resampling_factor
        resampled_indexer = np.arange(actual_start, actual_end, self._oversampling_period)

        self._values_interpolator = scipy.interpolate.interp1d(
            original_indexer,
            values,
            axis=0,
            kind=self.interpolation,
            bounds_error=False,
            fill_value="extrapolate",
        )

        # resample
        resampled_timestamps = resampled_indexer
        resampled_values = self.values_interpolator(resampled_indexer)

        # Quantisation
        core_count = len(
            self.config.environments_apps_zones[self.config.env]["src"]["app_config"][
                "generator"
            ].cores
        )
        thresholds = self.config.environments_apps_zones[self.config.env]["snk"][
            "zone_config"
        ].power_thresholds[rdpstream.columns[-1].split(":")[-2]]
        quantisation = self._apply_mapping(np.arange(len(thresholds)), core_count)
        for tidx in range(len(thresholds)):
            if tidx < len(thresholds) - 1:
                resampled_values[
                    np.logical_and(
                        resampled_values >= thresholds[tidx],
                        resampled_values < thresholds[tidx + 1],
                    )
                ] = quantisation[tidx]
            else:
                resampled_values[resampled_values >= thresholds[tidx]] = quantisation[tidx]

        # Fine Sync
        if self.sampling_period > self._oversampling_period:
            ideal_timestamps = np.hstack(
                [[0], np.cumsum(self.config.rdpstream.iloc[:, 0]).to_numpy()]
            )
            ideal_values = np.hstack(
                [
                    [self.config.rdpstream.iloc[0, 1]],
                    self.config.rdpstream.iloc[:, 1].to_numpy(),
                ]
            )
            resampled_ideal_timestamps = np.arange(
                ideal_timestamps[0], ideal_timestamps[-1], self._oversampling_period
            )
            ideal_values_interpolator = scipy.interpolate.interp1d(
                ideal_timestamps,
                ideal_values,
                axis=0,
                kind="next",
                bounds_error=False,
                fill_value="extrapolate",
            )

            resampled_ideal_values = ideal_values_interpolator(resampled_ideal_timestamps)

            num_idxes = int(50.0 * self._new_samples_per_symbol)
            # take the end of the trace....
            corr_start_idx = int(resampled_ideal_timestamps.size - num_idxes)
            corr_end_idx = int(resampled_ideal_timestamps.size - 1)

            crosscorr = np.correlate(
                resampled_ideal_values, resampled_values[corr_start_idx:corr_end_idx]
            )
            timediff = np.arange(
                0,
                np.diff(resampled_ideal_timestamps).mean() * crosscorr.size,
                np.diff(resampled_ideal_timestamps).mean(),
            )

            timediff_interval = np.where(timediff <= 0.1)[0][-1]
            time_offset = timediff[crosscorr.argmax()] - resampled_timestamps[corr_start_idx]
            idx_offset = int(time_offset // self._oversampling_period) * (-1)

            resampled_values = resampled_values[idx_offset:]
            resampled_timestamps = resampled_timestamps[idx_offset:]
        else:
            time_offset = 0

        # reshape
        reshaped_length = resampled_values.shape[0] // self._new_samples_per_symbol
        length_limit = reshaped_length * self._new_samples_per_symbol

        self._decode_params_ = {
            "actual_start": actual_start,
            "actual_end": actual_end,
            "duration": actual_end - actual_start,
            "sampling_period_inferred": sampling_period_inferred,
            "self.sampling_period": self.sampling_period,
            "self.symbol_rate": self.config.symbol_rate,
            "orig_samples_per_symbol": orig_samples_per_symbol,
            "subsymbol_count": subsymbol_count,
            "original_size": original_size,
            "reshaped_length": reshaped_length,
            "resampled_values": resampled_values.shape,
            "resampled_indexer": resampled_indexer.shape,
            "length_limit": length_limit,
            "self._resampling_factor": self._resampling_factor,
            "self._new_samples_per_symbol": self._new_samples_per_symbol,
            "self.interpolation": self.interpolation,
            "time_offset": time_offset,
        }

        self._decode_timestamps = resampled_timestamps[:length_limit].reshape(
            reshaped_length, self._new_samples_per_symbol
        )

        self.add_intermediate("slicing", self._decode_timestamps[:, 0])
        self.add_intermediate("timestamps", self._decode_timestamps)

        return resampled_values[:length_limit].reshape(
            reshaped_length, self._new_samples_per_symbol
        )
