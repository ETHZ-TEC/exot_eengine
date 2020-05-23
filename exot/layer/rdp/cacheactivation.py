"""Cache-activation RDP layer"""

# TODO: There's clearly room for deduplication and extraction of common functionality
#       between the core/cache activation layers and the prospective resampling
#       layers.

import typing as t

import numpy
import pandas
import scipy.interpolate
import scipy.signal

from exot.exceptions import *
from exot.util.misc import get_valid_access_paths, getitem, is_scalar_numeric

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


def row_mean(x: np.ndarray) -> float:
    return x.mean()


class CacheActivation(Layer, layer=Layer.Type.PrePost):
    def __init__(
        self,
        *,
        sampling_period: float,
        environments_apps_zones: t.Mapping,
        saturating: bool = True,
        reducer: t.Callable = row_mean,
        **kwargs,
    ):
        """Initialise the CacheActivation RDP layer

        Args:
            sampling_period (float): the sink app's sampling period
            environments_apps_zones (t.Mapping): the env->app->zone mapping
            saturating (bool, optional): is configured as saturating?
            reducer (t.Callable, optional): function that reduces rows to single column
            **kwargs: unused
        """
        self.sampling_period = sampling_period
        self.environments_apps_zones = environments_apps_zones
        self.saturating = saturating
        self.reducer = reducer
        self.cache_set_counts = get_cache_set_counts(self.environments_apps_zones)

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
            raise MisconfiguredError("subsymbol_rate must be a numeric value")
        if not is_scalar_numeric(self.config.symbol_rate):
            raise MisconfiguredError("symbol_rate must be a numeric value")

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
    def saturating(self):
        """Is the layer operating in the saturating mode?"""
        return self._saturating

    @saturating.setter
    def saturating(self, value):
        """Set the saturating property"""
        if not isinstance(value, (bool, np.bool)):
            raise LayerMisconfigured(f"'saturating' must be a boolean, got: {value}")

        self._saturating = value

    @property
    def timing_interpolator(self) -> t.Optional[t.Callable]:
        """Get the timing interpolator

        Returns:
            t.Optional[t.Callable]: the interpolator function, if available
        """
        return getattr(self, "_timing_interpolator", None)

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

    def _apply_mapping(self, stream: np.ndarray, set_count: int) -> np.ndarray:
        """Apply a cache set-specifier mapping to an input stream

        Args:
            stream (np.ndarray): the 1-d lnestream
            set_count (int): the cache set count

        Returns:
            np.ndarray: a validated and optionally saturated stream

        Raises:
            ValueValidationFailed: if any of the values is not within [0, 2^set_count -1]
        """

        # If operating in the 'saturating' mode
        if self.saturating:
            # If a negative value is encountered in the stream, replace it with the max,
            # which is (set_count), and subtract (value + 1). For example, with a cache set
            # count of 4, '-1' will yield '4', '-2' will yield '3', and so on.
            if (stream < 0).any():
                stream[stream < 0] = set_count - (stream[stream < 0] + 1)

            stream = (2 ** stream - 1).astype(int)

        _ = self._upper_limit(set_count)
        if (stream < 0).any() or (stream > _).any():
            raise ValueValidationFailed(
                f"some values in the mapped stream for set_count of {set_count} were "
                f"out of range [0, {_}]"
            )

        return stream

    def _encode(self, lnestream: np.ndarray) -> pd.DataFrame:
        """Encode a lnestream for each cache set/schedule pair

        Args:
            lnestream (np.ndarray): the line-encoded stream from an LNE coder

        Returns:
            pd.DataFrame: a DataFrame with a timestamps and value columns named after tags
        """
        tag_count = len(self.cache_set_counts)
        rdpstream = np.empty((lnestream.shape[0], tag_count), dtype=np.dtype("int"))
        timestamps = np.full(lnestream.shape[0], 1 / self.config.subsymbol_rate)
        tags = []

        for idx, (set_count, tag) in enumerate(self.cache_set_counts):
            tags.append(tag)
            rdpstream[:, idx] = self._apply_mapping(lnestream.copy(), set_count)

        return pd.DataFrame.join(
            pd.DataFrame(timestamps, columns=["timestamp"]),
            pd.DataFrame(rdpstream, columns=tags),
        )

    @property
    def _decode_params(self):
        return getattr(self, "_decode_params_", None)

    @property
    def decode_timestamps(self) -> np.ndarray:
        """Get the timestamps of the resampled and reshaped rdpstream

        Notes:
        -   For plotting, one might want to access the 0-th index of the returned array.

        Returns:
            np.ndarray: 2-d array of same size as the output of `decode`.
        """
        return getattr(self, "_decode_timestamps", None)

    def _decode(self, rdpstream: pd.DataFrame) -> np.ndarray:
        """Resample and reshape an input rdpstream

        Args:
            rdpstream (pd.DataFrame): the rdpstream DataFrame produced by the I/O layer

        Returns:
            np.ndarray: a resampled and reshaped array, of width that is a multiple of
            the subsymbol_count, in the range [2 × subsymbol_count, 4 × subsymbol_count]
        """
        timestamps = rdpstream.iloc[:, 0]
        values = rdpstream.iloc[:, 1:].apply(self.reducer, axis=1, result_type="expand")

        samples_per_symbol: float
        sampling_period_inferred = (
            0.15 * timestamps.diff().mean() + 0.85 * timestamps.diff().median()
        )

        samples_per_symbol = 1 / (sampling_period_inferred * self.config.symbol_rate)
        samples_per_symbol_ideal = 1 / (self.sampling_period * self.config.symbol_rate)

        extra_params = {}

        if "symstream" in self.config:
            actual_start = timestamps[0]
            actual_end = timestamps[len(timestamps) - 1]

            desired_len = len(self.config.symstream)
            ideal_end = (desired_len * samples_per_symbol_ideal) * self.sampling_period
            desired_end = (desired_len * samples_per_symbol - 1) * sampling_period_inferred

            idx = pd.Series.idxmax(timestamps >= (actual_start + desired_end))
            data_slice = slice(0, idx + 1)
            timestamps = timestamps[data_slice]
            values = values[data_slice]

            new_start = timestamps[0]
            new_end = timestamps[len(timestamps) - 1]

            extra_params.update(
                actual_start=actual_start,
                actual_end=actual_end,
                desired_len=desired_end,
                ideal_end=ideal_end,
                desired_end=desired_end,
                new_end=new_end,
            )

        subsymbol_count = self.config.subsymbol_rate / self.config.symbol_rate
        min_subsymbol_count = 2 * subsymbol_count

        if samples_per_symbol < min_subsymbol_count:
            self._samples_per_symbol = min_subsymbol_count
        elif samples_per_symbol > 4 * min_subsymbol_count:
            self._samples_per_symbol = 4 * min_subsymbol_count
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
        original_size = len(timestamps)
        resampled_size = np.ceil(original_size * self._resampling_factor)
        resampled_size = self._samples_per_symbol * np.floor(
            resampled_size // self._samples_per_symbol
        )

        if "symstream" in self.config:
            desired_len = len(self.config.symstream)
            if resampled_size % desired_len != 0:
                resampled_size = desired_len * np.round(resampled_size / desired_len)

        original_indexer = np.linspace(0, 1, original_size)
        resampled_indexer = np.linspace(0, 1, resampled_size)
        self._timing_interpolator = scipy.interpolate.interp1d(original_indexer, timestamps)
        self._values_interpolator = scipy.interpolate.interp1d(original_indexer, values)

        # resample
        resampled_timestamps = self.timing_interpolator(resampled_indexer)
        resampled_values = self.values_interpolator(resampled_indexer)

        # reshape
        reshaped_length = int(resampled_size / self._samples_per_symbol)
        length_limit = int(resampled_size)

        self._decode_params_ = {
            "sampling_period_inferred": sampling_period_inferred,
            "self.sampling_period": self.sampling_period,
            "self.symbol_rate": self.config.symbol_rate,
            "samples_per_symbol": samples_per_symbol,
            "samples_per_symbol_ideal": samples_per_symbol_ideal,
            "subsymbol_count": subsymbol_count,
            "min_subsymbol_count": min_subsymbol_count,
            "original_size": original_size,
            "resampled_size": resampled_size,
            "reshaped_length": reshaped_length,
            "length_limit": length_limit,
            "self._resampling_factor": self._resampling_factor,
            "self._samples_per_symbol": self._samples_per_symbol,
        }

        self._decode_params_.update(**extra_params)

        self._decode_timestamps = resampled_timestamps[:length_limit].reshape(
            reshaped_length, self._samples_per_symbol
        )

        return resampled_values[:length_limit].reshape(
            reshaped_length, self._samples_per_symbol
        )
