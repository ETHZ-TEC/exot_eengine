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
import typing as t

import numpy
import pandas
import scipy.interpolate
import scipy.signal

from exot.exceptions import *
from exot.util.misc import get_valid_access_paths, getitem, is_scalar_numeric
from exot.util.scinum import find_ramp_edges, interleave

from .._base import Layer

np = numpy
pd = pandas


def get_cache_set_counts(environments_apps_zones: t.Mapping) -> set:
    """Extracts the set_count value from experiment app configurations

    Args:
        environments_apps_zones (t.Mapping): The mapping env -> app -> zone

    Returns:
        set: A set with (set-count, schedule-tag) pairs

    Raises:
        LayerMisconfigured: If field not found in the config.
    """
    e_a_z = environments_apps_zones
    _cache_set_counts = set()

    for env in e_a_z:
        for app in e_a_z[env]:
            if app != "src":
                continue

            _path_to_set_count = ("app_config", "generator", "set_count")
            _path_to_schedule_tag = ("zone_config", "schedule_tag")

            access_paths = list(get_valid_access_paths(e_a_z[env][app]))

            if _path_to_set_count not in access_paths:
                raise LayerMisconfigured(
                    f"{env!r}->{app!r} must have a 'generator.cache_set_count' config key"
                )
            if _path_to_schedule_tag not in access_paths:
                _ = e_a_z[env][app]["zone"]
                raise LayerMisconfigured(
                    f"{env!r}.{_!r} of app {app!r} must have a schedule_tag"
                )

            _cache_set_counts.add(
                (
                    getitem(e_a_z[env][app], _path_to_set_count),
                    getitem(e_a_z[env][app], _path_to_schedule_tag),
                )
            )

    return _cache_set_counts


class DirectActivation(Layer, layer=Layer.Type.PrePost):
    def __init__(
        self,
        *,
        sampling_period: float,
        environments_apps_zones: t.Mapping,
        sync_pulse_duration: float = 0.0,
        sync_pulse_detection: str = "falling",
        **kwargs,
    ):
        """Initialise the DirectActivation RDP layer

        Args:
            sampling_period (float): the sink app's sampling period
            environments_apps_zones (t.Mapping): the env->app->zone mapping
            sync_pulse_duration (float, optional): Duration of the sync pulse
            sync_pulse_detection (str, optional): Which edge to look first for?
            **kwargs: unused
        """
        self.sampling_period = sampling_period
        self.environments_apps_zones = environments_apps_zones
        self.sync_pulse_duration = sync_pulse_duration
        self.sync_pulse_detection = sync_pulse_detection
        self.cache_set_counts = get_cache_set_counts(self.environments_apps_zones)

    @property
    def sync_pulse_detection(self):
        return self._sync_pulse_detection

    @sync_pulse_detection.setter
    def sync_pulse_detection(self, value):
        if not isinstance(value, str):
            raise TypeError("'sync_pulse_detection' must be a string")
        if value not in ["falling", "rising"]:
            raise ValueError("'sync_pulse_detection' must be either 'falling' or 'rising'")

        self._sync_pulse_detection = value

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
                lambda v: v.shape[1] == 1 + len(self.cache_set_counts),
            ]
        }

    @property
    def sync_pulse_duration(self):
        return getattr(self, "_sync_pulse_duration", 0.0)

    @sync_pulse_duration.setter
    def sync_pulse_duration(self, value):
        if not is_scalar_numeric(value):
            raise LayerMisconfigured("sync pulse duration must be a scalar numeric value")
        if value < 0.0:
            raise LayerMisconfigured("sync pulse duration must be 0 or positive")

        self._sync_pulse_duration = value

    @property
    def _decode_validators(self):
        # Accepts: timestamps + values for each configured cache set
        return {
            pd.DataFrame: [
                lambda v: v.ndim == 2,
                lambda v: v.shape[1] in map(lambda x: 1 + x[0], self.cache_set_counts),
            ]
        }

    @property
    def _decode_output_validators(self):
        return {pd.DataFrame: [lambda v: v.ndim == 2, lambda v: v.shape[1] >= 2]}

    @property
    def requires_runtime_config(self) -> (bool, bool):
        """Does the layer's (encode, decode) require runtime configuration?"""
        return (True, True)

    @property
    def required_config_keys(self):
        """The required config keys

        Implements the `required_config_keys` from Configurable base class
        """
        return ["symbol_rate", "subsymbol_rate"]

    def validate(self) -> t.NoReturn:
        """Implementation of Configurable's `validate`"""
        if not is_scalar_numeric(self.config.subsymbol_rate):
            raise LayerMisconfigured("subsymbol_rate must be a numeric value")
        if not is_scalar_numeric(self.config.symbol_rate):
            raise LayerMisconfigured("symbol_rate must be a numeric value")

    @property
    def sampling_period(self):
        """Get the sampling period"""
        return self._sampling_period

    @sampling_period.setter
    def sampling_period(self, value):
        """Set the sampling period"""
        if not is_scalar_numeric(value):
            raise LayerMisconfigured("sampling_period must be an integer of float")
        self._sampling_period = value

    @property
    def cache_set_counts(self):
        """Get the cache set count and schedules set of 2-tuples"""
        return self._cache_set_counts

    @cache_set_counts.setter
    def cache_set_counts(self, value):
        """Set the cache set count and schedules set of 2-tuples"""
        try:
            assert isinstance(value, set), "must be a set"
            all(
                isinstance(_, tuple) and len(_) == 2 for _ in value
            ), "must contain only 2-tuples"
            all(
                isinstance(_[0], int) and isinstance(_[1], str) for _ in value
            ), "must contain 2-tuples of (int, str)"
        except AssertionError as e:
            raise LayerMisconfigured("cache_set_counts {}, got {}".format(*e.args, value))

        self._cache_set_counts = value

    @property
    def values_interpolator(self) -> t.Optional[t.Callable]:
        """Get the values interpolator

        Returns:
            t.Optional[t.Callable]: the interpolator function, if available
        """
        return getattr(self, "_values_interpolator", None)

    @staticmethod
    def _upper_limit(set_count: np.integer) -> np.integer:
        """Get the upper limit for valid cache set specifiers

        Args:
            set_count (np.integer): the cache set count

        Returns:
            np.integer: the upper limit
        """
        assert isinstance(set_count, (int, np.integer)), "set_count must be an int"
        return 2 ** set_count - 1

    def _apply_encoding(self, stream: np.ndarray, set_count: int) -> np.ndarray:
        """Apply a cache set-specifier mapping to an input stream

        Args:
            stream (np.ndarray): the 1-d lnestream
            set_count (int): the cache set count

        Returns:
            np.ndarray: a validated stream

        Raises:
            ValueValidationFailed: if any of the values is not within [0, 2^set_count -1]
        """

        upper_limit = self._upper_limit(set_count)
        if (stream < 0).any() or (stream > upper_limit).any():
            raise ValueValidationFailed(
                f"some values in the mapped stream for set_count of {set_count} were "
                f"out of range [0, {upper_limit}]"
            )

        return stream

    @staticmethod
    def _insert_sync(
        timestamps: np.ndarray, rdpstream: np.ndarray, duration: float
    ) -> (np.ndarray, np.ndarray):
        _ = rdpstream.shape[1]
        _type = rdpstream.dtype.type
        return (
            np.insert(timestamps, 0, 3 * [duration]),
            np.vstack([np.array([_ * [_type(0)], _ * [_type(1)], _ * [_type(0)]]), rdpstream]),
        )

    def _encode(self, lnestream: np.ndarray) -> pd.DataFrame:
        """Encode a lnestream for each cache set/schedule pair

        Args:
            lnestream (np.ndarray): the line-encoded stream from an LNE coder

        Returns:
            pd.DataFrame: a DataFrame with a timestamps and value columns named after tags
        """
        tag_count = len(self.cache_set_counts)
        rdpstream = np.empty((lnestream.shape[0], tag_count), dtype=np.uint64)
        timestamps = np.full(lnestream.shape[0], 1 / self.config.subsymbol_rate)
        tags = []

        for idx, (set_count, tag) in enumerate(self.cache_set_counts):
            tags.append(tag)
            rdpstream[:, idx] = self._apply_encoding(lnestream.copy(), set_count)

        if self.sync_pulse_duration > 0.0:
            timestamps, rdpstream = self._insert_sync(
                timestamps, rdpstream, self.sync_pulse_duration
            )

        return pd.DataFrame.join(
            pd.DataFrame(timestamps, columns=["timestamp"]),
            pd.DataFrame(rdpstream, columns=tags),
        )

    @property
    def _decode_params(self):
        return getattr(self, "_decode_params_", None)

    @property
    def decode_timestamps(self) -> np.ndarray:
        """Get the timestamps of the resampled and reshaped rdpstream"""
        return getattr(self, "_decode_timestamps", None)

    def _decode(self, rdpstream: pd.DataFrame) -> np.ndarray:
        """Resample and reshape an input rdpstream"""
        timestamps = pandas.Series(rdpstream.iloc[:, 0])
        values = rdpstream.iloc[:, 1:]
        set_count = values.shape[1]

        _allowed = [count for count, tag in self.cache_set_counts]
        if set_count not in _allowed:
            raise LayerRuntimeError(
                "Unexpected set count ({}) in rdpstream, expected one of {!r}".format(
                    set_count, _allowed
                )
            )

        actual_start = timestamps.iloc[0]
        actual_end = timestamps.iloc[len(timestamps) - 1]
        original_size = len(timestamps)

        sampling_period = self.config.get("sampling_period", self.sampling_period)
        samples_per_symbol = 1 / (sampling_period * self.config.symbol_rate)
        subsymbol_count = self.config.subsymbol_rate / self.config.symbol_rate
        min_subsymbol_count = 2 * subsymbol_count

        if samples_per_symbol < min_subsymbol_count:
            self._samples_per_symbol = min_subsymbol_count
        elif samples_per_symbol > 1024 * min_subsymbol_count:
            self._samples_per_symbol = 1024 * min_subsymbol_count
        else:
            self._samples_per_symbol = samples_per_symbol

        # make sure that the _samples_per_symbol is a multiple of min_subsymbol_count
        if self._samples_per_symbol % min_subsymbol_count != 0:
            self._samples_per_symbol = min_subsymbol_count * np.ceil(
                self._samples_per_symbol / min_subsymbol_count
            )

        # make sure that _samples_per_symbol is an integer
        if not self._samples_per_symbol.is_integer():
            self._samples_per_symbol = np.ceil(self._samples_per_symbol)
        else:
            self._samples_per_symbol = int(self._samples_per_symbol)

        self._resampling_factor = self._samples_per_symbol / samples_per_symbol

        # set-up resampling
        original_indexer = timestamps.to_numpy()

        self._oversampling_period = sampling_period / self._resampling_factor
        round_off_fix = 1e3
        resampled_indexer = (
            np.arange(
                actual_start * round_off_fix,
                actual_end * round_off_fix,
                self._oversampling_period * round_off_fix,
            )
            / round_off_fix
        )

        self._values_interpolator = scipy.interpolate.interp1d(
            original_indexer,
            values,
            axis=0,
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )

        # resample
        resampled_timestamps = pd.Series(resampled_indexer)
        resampled_values = self.values_interpolator(resampled_indexer)

        self._decode_params_ = {
            "set_count": set_count,
            "self.sampling_period": self.sampling_period,
            "sampling_period": sampling_period,
            "self.symbol_rate": self.config.symbol_rate,
            "samples_per_symbol": samples_per_symbol,
            "subsymbol_count": subsymbol_count,
            "min_subsymbol_count": min_subsymbol_count,
            "original_size": original_size,
            "resampled_timestamps.shape": resampled_timestamps.shape,
            "resampled_values.shape": resampled_values.shape,
            "self._resampling_factor": self._resampling_factor,
            "self._samples_per_symbol": self._samples_per_symbol,
            "self._oversampling_period": self._oversampling_period,
        }

        extra_params = {}

        self._debug_values = resampled_values.copy()

        # if sync pulse duration is greater than 0, perform pulse detection and slice
        if self.sync_pulse_duration > 0.0:
            whole_pulse = self.sync_pulse_duration * 3
            mid_pulse = self.sync_pulse_duration / 5
            search_start = actual_start + mid_pulse
            search_end = actual_start + whole_pulse

            _origin = None

            if "find_edges" in self.config:
                if self.config.find_edges is False:
                    _origin = pd.Series.idxmax(
                        resampled_timestamps > actual_start + 3 * self.sync_pulse_duration
                    )

            # look for edges in the interval [search_start, search_end]
            idx_start = pd.Series.idxmax(resampled_timestamps >= search_start)
            idx_end = pd.Series.idxmax(resampled_timestamps >= search_end)
            falling, rising, fc, rc, convolved, *other = find_ramp_edges(
                resampled_values[idx_start : idx_end + 1, 0],
                roll=self.config.roll if "roll" in self.config else 5,
                roll_fun=self.config.roll_fun if "roll_fun" in self.config else np.median,
                threshold=self.config.threshold if "threshold" in self.config else 0.5,
                _debug=True,
            )

            falling += idx_start
            rising += idx_start

            # if detecting a first falling, then rising edge, the edges are reversed
            if self.sync_pulse_detection == "falling":
                rising, falling = falling, rising

            extra_params.update(search_end=search_end, falling=falling, rising=rising)

            _adjust = self.config.adjust if "adjust" in self.config else -1
            _thresh = self.config.threshold if "threshold" in self.config else 0.985

            _chosen_redge = None
            _chosen_fedge = None

            _iterations_to_find_edge = 0

            _lower_lim = _thresh * self.sync_pulse_duration
            _upper_lim = (2 - _thresh) * self.sync_pulse_duration

            for redge in rising:
                if _origin is not None:
                    break

                for fedge in falling:
                    _iterations_to_find_edge += 1
                    duration = resampled_timestamps[fedge] - resampled_timestamps[redge]

                    if (duration >= _lower_lim) and (duration <= _upper_lim):
                        _origin = (
                            pd.Series.idxmax(
                                resampled_timestamps
                                > resampled_timestamps[fedge] + self.sync_pulse_duration
                            )
                            # + _adjust
                        )

                        _chosen_redge = resampled_timestamps[redge]
                        _chosen_fedge = resampled_timestamps[fedge]

                        break

            # if origin finding fails, make a guess
            if _origin is None:
                # add debug intermediate
                _debug = pandas.DataFrame()
                _debug["falling"] = [falling]
                _debug["rising"] = [rising]
                _debug["timestamps"] = [resampled_timestamps]
                _debug["data"] = [self._debug_values[:, 0]]
                _debug["idx_start_end"] = [np.array([idx_start, idx_end])]
                _debug["convolved"] = [convolved]
                self.add_intermediate("edge_detection_debug", _debug)

                raise LayerRuntimeError("Could not find the sync pulse!")

            _origin = _origin + _adjust

            extra_params.update(
                idx_start=idx_start,
                idx_end=idx_end,
                _origin=_origin,
                _origin_time=resampled_timestamps[_origin],
                _chosen_redge=_chosen_redge,
                _chosen_fedge=_chosen_fedge,
                _iterations_to_find_edge=_iterations_to_find_edge,
            )

            sync_slice = slice(_origin, None)

            self.add_intermediate(
                "edge_detection",
                np.array([resampled_timestamps[_origin], _chosen_fedge, _chosen_redge]),
            )

            resampled_timestamps = resampled_timestamps[sync_slice]
            resampled_values = resampled_values[sync_slice]

        self._decode_params_.update(**extra_params)

        length_limit = self._samples_per_symbol * (
            resampled_values.shape[0] // self._samples_per_symbol
        )
        resize_slice = slice(None, length_limit)

        resampled_values = resampled_values[resize_slice]
        resampled_timestamps = resampled_timestamps[resize_slice]

        self._debug_timestamps = resampled_timestamps

        # reorder as:
        decode_values = interleave(resampled_values, n=self._samples_per_symbol)
        decode_values = np.array(np.split(decode_values, set_count, axis=1)).transpose(1, 0, 2)

        _ = np.repeat(
            np.array(np.split(resampled_timestamps, decode_values.shape[0])), set_count, axis=0
        )
        self._decode_timestamps = _.reshape(*decode_values.shape)
        self.add_intermediate("slicing", self._decode_timestamps[:, 0, 0])
        self.add_intermediate("timestamps", self._decode_timestamps)

        return decode_values
