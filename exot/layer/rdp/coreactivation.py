"""Core-activation RDP layer"""

import copy
import typing as t

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.signal

from exot.exceptions import *
from exot.util.misc import get_valid_access_paths, getitem, is_scalar_numeric, get_cores_and_schedules

from .._mixins import RDPmixins
from .._base import Layer

"""
CoreActivation
--------------

The CoreActivation performs the following:

-   [__Encode__]: takes a 1-d line-encoded stream, creates timestamps using a
    `subsymbol_rate` provided during execution, and produces a DataFrame with
    a `timestamp` as the 1st column, followed by columns with post-processed streams
    for each unique combination of `core_count` and `schedule_tag` for all configured
    environments/apps/zones.

    The post-processing maps line-encoded symbols using a symbol -> core-specifier
    mapping. The core-specifier is simply a the symbol as an integer represeted as a
    binary number.

    The `saturating` parameter is used to configure the mode in which the layer
    operates: for example, in saturating mode, the value of '2' is interpreted as
    "2 active", resulting in a core-specifier 0b11, which is equal to 3. The saturating
    operation is (2^x - 1), which produces binary numbers with all 1's (except for input
    of '0'). All produced values are in the range [0, 2^cores - 1].

    In non-saturating mode the values are simply put through. Negative input values of are
    transformed into the (maximum saturating value + 1 - value). Checks are performed to
    make sure that all values lie in the valid range.

    Preconditions:
    -   Input is a 1-d array

    Postconditions:
    -   Output is a 2-d DataFrame with timestamps as 1st column,

-   [__Decode__]: takes a DataFrame with timestamps and values, oversamples the values in
    order to produce 2-d array of specific width (at least 2 × subsymbol count of the
    following line-coding layer), and outputs the reshaped values. After decoding, the
    corresponding timestamps array is available via the `decode_timestamps` property.

    Preconditions:
    -   Input is a 2-d DataFrame
    -   Input DataFrame has exactly 2 columns (timestamps + values)

    Postconditions:
    -   Output is a NumPy array
    -   Output is at least twice and less than 4 times the subsymbol count


TODO: This layer could inherit from a generic output resampling layer, and apply own
      encoding method. Refactoring might be beneficial.
"""


class CoreActivation(RDPmixins, Layer, layer=Layer.Type.PrePost):
    def __init__(
        self,
        *,
        sampling_period: float,
        environments_apps_zones: t.Mapping,
        saturating: bool = True,
        interpolation: str = "linear",
        **kwargs,
    ):
        """Initialise the CoreActivation RDP layer

        Args:
            sampling_period (float): the sink app's sampling period
            environments_apps_zones (t.Mapping): the env->app->zone mapping
            saturating (bool, optional): is configured as saturating?
        """
        self.interpolation = interpolation
        self.sampling_period = sampling_period
        self.environments_apps_zones = environments_apps_zones
        self.saturating = saturating
        self.cores_and_schedules = get_cores_and_schedules(self.environments_apps_zones)

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
        # Accepts: timestamps + single value
        return {pd.DataFrame: [lambda v: v.ndim == 2, lambda v: v.shape[1] == 2]}

    @property
    def _decode_output_validators(self):
        # Accepts: timestamps + single value
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
    def _upper_limit(core_count: np.integer) -> np.integer:
        """Get the upper limit for valid core specifiers

        Args:
            core_count (np.integer): the core count

        Returns:
            np.integer: the upper limit
        """
        assert isinstance(core_count, (int, np.integer)), "core_count must be an int"
        return 2 ** core_count - 1

    def _apply_mapping(self, stream: np.ndarray, core_count: int) -> np.ndarray:
        """Apply a core-specifier mapping to an input stream

        Args:
            stream (np.ndarray): the 1-d lnestream
            core_count (int): the core count

        Returns:
            np.ndarray: a validated and optionally saturated stream

        Raises:
            ValueValidationFailed: if any of the values is not within [0, 2^core_count -1]
        """

        # If operating in the 'saturating' mode
        if self.saturating:
            # If a negative value is encountered in the stream, replace it with the max,
            # which is (core_count), and subtract (value + 1). For example, with a core
            # count of 4, '-1' will yield '4', '-2' will yield '3', and so on.
            if (stream < 0).any():
                stream[stream < 0] = core_count - (stream[stream < 0] + 1)

            stream = (2 ** stream - 1).astype(int)

        _ = self._upper_limit(core_count)
        if (stream < 0).any() or (stream > _).any():
            raise ValueValidationFailed(
                f"some values in the mapped stream for core_count of {core_count} were "
                f"out of range [0, {_}]"
            )

        return stream

    def _encode(self, lnestream: np.ndarray) -> pd.DataFrame:
        """Encode a lnestream for each core/schedule pair

        Args:
            lnestream (np.ndarray): the line-encoded stream from an LNE coder

        Returns:
            pd.DataFrame: a DataFrame with a timestamps and value columns named after tags
        """
        tag_count = len(self.cores_and_schedules)
        rdpstream = np.empty((lnestream.shape[0], tag_count), dtype=np.dtype("int"))
        timestamps = np.full(lnestream.shape[0], 1 / self.config.subsymbol_rate)
        tags = []

        for idx, (core_count, tag) in enumerate(self.cores_and_schedules):
            tags.append(tag)
            rdpstream[:, idx] = self._apply_mapping(lnestream.copy(), core_count)

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
        values = rdpstream.iloc[:, 1]

        actual_start = timestamps.iloc[0]
        actual_end = timestamps.iloc[len(timestamps) - 1]

        orig_samples_per_symbol: float
        sampling_period_inferred = (
            0.15 * timestamps.diff().mean() + 0.85 * timestamps.diff().median()
        )

        if abs(sampling_period_inferred - self.sampling_period)  / self.sampling_period > 0.1:
            pass
            # TODO here we need a logger output to tell the user somethings off with the timestamps

        orig_samples_per_symbol      = 1 / (self.sampling_period * self.config.symbol_rate)
        subsymbol_count              = self.config.subsymbol_rate / self.config.symbol_rate
        self._new_samples_per_symbol = max([subsymbol_count * 100, orig_samples_per_symbol])
        #self._new_samples_per_symbol = orig_samples_per_symbol

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
            "self.interpolation":self.interpolation,
#            "time_offset":time_offset,
        }

        self._decode_timestamps = resampled_timestamps[:length_limit].reshape(
            reshaped_length, self._new_samples_per_symbol
        )

        self.add_intermediate("slicing", self._decode_timestamps[:, 0])
        self.add_intermediate("timestamps", self._decode_timestamps)

        return resampled_values[:length_limit].reshape(
            reshaped_length, self._new_samples_per_symbol
        )
