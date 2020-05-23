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
"""Scientific and numeric helpers

Attributes:
    POWERS_OF_2 (np.ndarray): The 64 consecutive powers of 2 (from 0 to 63)
"""

import typing as t

import numba
import numpy as np
import pandas
import scipy
import scipy.signal
import sklearn.base
import sklearn.exceptions
import sklearn.utils.validation

from .misc import is_scalar_numeric

__all__ = (
    "add_awgn_noise",
    "add_uniform_noise",
    "awgn_like",
    "count_errors",
    "count_errors_robust",
    "find_ramp_edges",
    "get_fft",
    "get_nearest",
    "get_nearest_index",
    "get_nearest_value",
    "get_welch",
    "interleave",
    "interleave_split",
    "is_binary",
    "is_fitted",
    "pack_array",
    "pack_bits",
    "pack_bits_fast",
    "pad_split",
    "uniform_like",
    "unpack_array",
    "unpack_bits",
    "unpack_bits_fast",
)


def get_nearest(array: t.Union[t.List, np.ndarray, pandas.Series], value: t.Any) -> tuple:
    """Gets the index and value of an element in the array that is closest to the given value

    Args:
        array (t.Union[t.List, np.ndarray, pandas.Series]): The array
        value (t.Any): The value

    Returns:
        tuple: A 2-tuple with the index and the value

    Raises:
        TypeError: Wrong array type provided
        ValueError: Array with wrong dimensions provided
        TypeError: Provided value is not a scalar numeric value
    """
    valid_types = (t.List, np.ndarray, pandas.Series)

    if not isinstance(array, valid_types):
        raise TypeError(f"'array' must be one of {valid_types!r}")

    if isinstance(array, t.List):
        array = np.asarray(array)
    else:
        if np.sum(array.shape) != array.size:
            raise ValueError("only 1-d or column arrays are allowed")

    if not is_scalar_numeric(value):
        raise TypeError("'value' must be a scalar numeric type")

    index = np.abs(array - value).idxmin()
    value = array[index]

    return (index, value)


def get_nearest_value(array: t.Union[t.List, np.ndarray, pandas.Series], value: t.Any) -> t.Any:
    """Gets the value of an element in the array that is closest to the given value

    Args:
        array (t.Union[t.List, np.ndarray, pandas.Series]): The array
        value (t.Any): The value

    Returns:
        t.Any: The nearest value
    """
    return get_nearest(array, value)[1]


def get_nearest_index(array: t.Union[t.List, np.ndarray, pandas.Series], value: t.Any) -> int:
    """Gets the index of an element in the array that is closest to the given value

    Args:
        array (t.Union[t.List, np.ndarray, pandas.Series]): [description]
        value (t.Any): [description]

    Returns:
        int: [description]
    """
    return get_nearest(array, value)[0]


def awgn_like(like: np.ndarray, sigma: float = 1.0, mu: float = 0.0) -> np.ndarray:
    """Produces a gaussian-distributed random array of same shape as a given array

    Args:
        like (np.ndarray): The array (only shape considered)

    Keyword Args:
        sigma (float): The standard deviation of the normal distribution (default: {1.0})
        mu (float): The mean of the distribution (default: {0.0})

    Returns:
        np.ndarray: The AWGN array
    """
    return np.random.normal(loc=mu, scale=sigma, size=like.shape)


def uniform_like(like: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Produce a uniform-distributed random array of same shape as a given array

    Args:
        like (np.ndarray): The array (only shape considered)

    Keyword Args:
        low (float): The lower limit of the uniform distribution (default: {0.0})
        high (float): The upper limit of the uniform distribution (default: {1.0})

    Returns:
        np.ndarray: The uniform-distributed random array
    """
    return np.random.uniform(low=low, high=high, size=like.shape)


def add_awgn_noise(to: np.ndarray, sigma: float = 1.0, mu: float = 0.0) -> np.ndarray:
    """Adds gaussian noise to an array

    Args:
        to (np.ndarray): The array to which the noise will be added

    Keyword Args:
        sigma (float): The standard deviation of the normal distribution (default: {1.0})
        mu (float): The mean of the distribution (default: {0.0})

    Returns:
        np.ndarray: The given array with added noise
    """
    return to + awgn_like(to, sigma=sigma, mu=mu)


def add_uniform_noise(to: np.ndarray, low: float = 0.0, high: float = 1.0):
    """Adds uniform noise to an array
        Args:
        to (np.ndarray): The array to which the noise will be added

    Keyword Args:
        low (float): The lower limit of the uniform distribution (default: {0.0})
        high (float): The upper limit of the uniform distribution (default: {1.0})

    Returns:
        np.ndarray: The given array with added noise
    """
    return to + uniform_like(to, low=low, high=high)


def count_errors(left: np.ndarray, right: np.ndarray) -> int:
    """Counts differences between two arrays of same shape

    Args:
        left (np.ndarray): The left array
        right (np.ndarray): The right array

    Returns:
        int: The number of errors/differences

    Raises:
        ValueError: The arrays are of different shape
    """
    assert (
        left.shape == right.shape
    ), f"arrays must be of same shape, got: {left.shape} and {right.shape}"
    return np.count_nonzero((left != right).astype(int))


def count_errors_robust(left: np.ndarray, right: np.ndarray) -> int:
    """Count differences between two row or column vectors

    Args:
        left (np.ndarray): The left array
        right (np.ndarray): The right array

    Returns:
        int: The number of element differences and/or length differences

    Raises:
        ValueError: Wrong array dimensions provided
    """
    assert (
        left.ndim == right.ndim
    ), f"arrays must be of same dimensions, got: {left.ndim} and {right.ndim}"
    assert left.ndim <= 2, "must be at most 2-d"

    if left.ndim == 2:
        assert left.shape[1] == 1, "must be a single column"
        assert right.shape[1] == 1, "must be a single column"

    diff = abs(left.size - right.size)
    errors = np.count_nonzero((left[: right.size] != right[: left.size]).astype(int))

    return errors + diff


def is_fitted(estimator: sklearn.base.BaseEstimator) -> bool:
    """Is an estimator fitted?

    Args:
        estimator (sklearn.base.BaseEstimator): the estimator to check

    Returns:
        bool: True if fitted, False otherwise
    """
    if hasattr(estimator, "_is_fitted"):
        return estimator._is_fitted()

    try:
        sklearn.utils.validation.check_is_fitted(
            estimator,
            [
                "coef_",
                "classes_",
                "estimator_",
                "tree_",
                "coefs_",
                "cluster_centers_",
                "support_",
            ],
            all_or_any=any,
        )
        return True
    except sklearn.exceptions.NotFittedError:
        return False


def interleave(array: np.ndarray, n: int) -> np.ndarray:
    """Interleaves rows of an array

    Args:
        array (np.ndarray): The array to interleave
        n (int): The number of consecutive rows to interleave

    Returns:
        np.ndarray: The interleaved array of shape (array[0] / n, array[1] * n)

    Details:
        The interleaving is as follows:

        Given a 4x4 array:
        >>> array = np.arange(0, 16).reshape(4, 4)
        >>> array
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11],
               [12, 13, 14, 15]])

        Interleaving with n=2 will produce a 8x2 array:
        >>> interleave(array, 2)
        array([[ 0,  4,  1,  5,  2,  6,  3,  7],
               [ 8, 12,  9, 13, 10, 14, 11, 15]])
    """
    assert (
        array.shape[0] % n == 0
    ), f"array shape[0] ({array.shape[0]}) must be a multiple of n ({n})"
    assert n > 0, "interleaving factor must be greater than 0"

    result = np.empty((array.shape[0] // n, array.shape[1] * n), dtype=array.dtype)

    for i in range(0, n):
        result[:, i::n] = array[i::n]

    return result


def interleave_split(array: np.ndarray, n: int) -> np.ndarray:
    """Interleaves rows of an array and splits into multiple arrays

    Args:
        array (np.ndarray): The array to interleave and split
        n (int): The number of consecutive rows to interleave

    Returns:
        np.ndarray: The interleaved and split array
    """
    return np.split(interleave(array, n), n)


def is_binary(array: np.ndarray) -> bool:
    """Checks if input array is binary

    Args:
        array (np.ndarray): The array

    Returns:
        bool: True if contains only 0s and 1s
    """
    return np.isin(array, np.array([0, 1])).all()


POWERS_OF_2 = np.left_shift(1, np.arange(64, dtype=np.uint64)).astype(
    np.uint64
)  # type: np.ndarray


def pack_bits(array: np.ndarray) -> np.uint64:
    """'Packs' a binary array into an unsigned integer

    Similar to np.packbits, but packs up to uint64, not uint8.

    Args:
        array (np.ndarray): The binary array to 'pack'

    Returns:
        np.uint64: The packed array
    """
    assert is_binary(array), "must be a binary array of integers"
    assert array.size <= 64, "must contain at most 64 elements"

    return array[::-1].astype(np.uint).dot(POWERS_OF_2[: array.size]).astype(np.uint)


def pack_bits_fast(array: np.ndarray) -> np.uint64:
    """'Packs' a binary array into an unsigned integer without value/type checks

    Similar to np.packbits, but packs up to uint64, not uint8.

    Args:
        array (np.ndarray): The binary array to 'pack'

    Returns:
        np.uint64: The packed array
    """
    return array[::-1].astype(np.uint).dot(POWERS_OF_2[: array.size]).astype(np.uint)


def pad_split(array: np.ndarray, n: np.uint, *, pad="msb") -> np.ndarray:
    """Split a binary array int fixed-sized chunks while 0-padding unequal splits

    Args:
        array (np.ndarray): The binary array
        n (np.uint): chunk size

    Returns:
        np.ndarray: The split (and padded) array of shape (ceil(array.size/n), n)
    """
    assert is_binary(array), "must be a binary array of integers"
    assert array.ndim == 1, "must be a flat array"
    assert len(array), "must not be empty"
    assert isinstance(n, (int, np.uint)), "must be an unsigned integer"
    assert n > 0, "must be greater than zero"

    # get equal divisions and remainder
    equal_divisions, remainder = np.divmod(array.size, n)
    equal_length = int(equal_divisions * n)

    padding: tuple

    if pad == "msb":
        padding = (int(n - remainder), 0)
    elif pad == "lsb":
        padding = (0, int(n - remainder))
    else:
        raise ValueError(
            "pad_split's 'pad' keyword argument accepts "
            "either 'msb' or 'lsb', got: {!r}".format(pad)
        )

    if equal_divisions == 0:
        return np.vstack(
            [np.pad(array[equal_length:], padding, "constant", constant_values=np.array([0]))]
        )

    elif not remainder:
        return np.array(np.split(array, equal_divisions))
    else:
        equal_splitted = np.array(np.split(array[:equal_length], equal_divisions))
        padded_remainder = np.pad(
            array[equal_length:], padding, "constant", constant_values=np.array([0])
        )

        return np.vstack([equal_splitted, padded_remainder])


def pack_array(array: np.ndarray, n: np.uint, *, pad="msb") -> np.ndarray:
    """Packs a binary array into unsigned integers of bit width n

    Args:
        array (np.ndarray): The binary array to pack
        n (np.uint): The bit width / chunk size

    Returns:
        np.ndarray: The packed array of size np.ceil(array.size / n)
    """
    assert array.ndim == 1, "must be a flat array"
    assert n > 0, "must be greater than zero"
    assert n <= 64, "must be smaller than 64 (uint64 length)"

    splitted = pad_split(array, n)
    return np.apply_along_axis(pack_bits_fast, 1, splitted)


def unpack_bits(value: np.uint, n: np.uint = 64) -> np.ndarray:
    """Unpacks an unsigned integer into an array of bits

    Args:
        value (np.uint): The value to unpack
        n (np.uint, optional): The number of bits

    Returns:
        np.ndarray: The value unpacked into a binary array

    Raises:
        ValueError: Value cannot be unpacked because it cannot be represented with n bits
    """
    assert isinstance(value, (np.uint)), "must be an unsigned integer"
    assert n > 0, "must be greater than zero"
    assert n <= 64, "must be smaller than 64 (uint64 length)"

    if value >= 2 ** n:
        raise ValueError(
            "Cannot unpack the value '{}': aliasing would occur because it "
            "is greater than 2^{}-1 and cannot be represented with {} bits.".format(value, n, n)
        )

    return np.unpackbits(np.array([value], dtype=">u8").view(np.uint8))[64 - n :]


def unpack_bits_fast(value: np.uint, n: np.uint = 64) -> np.ndarray:
    """Unpacks an unsigned integer into an array of bits without value/type checks

    Args:
        value (np.uint): The value to unpack
        n (np.uint, optional): The number of bits

    Returns:
        np.ndarray: The value unpacked into a binary array
    """
    return np.unpackbits(np.array([value], dtype=">u8").view(np.uint8))[
        slice(int(64 - n), None)
    ]


def unpack_array(array: np.ndarray, n: np.uint) -> np.ndarray:
    """Unpacks an array of unsigned integers into a 2-d array of bits

    Args:
        array (np.ndarray): The array to unpack
        n (np.uint): The number of bits

    Returns:
        np.ndarray: The unpacked array of shape (array.size, n)

    Raises:
        ValueError: An array of unsuitable shape is passed
    """
    assert n <= 64, "must be smaller than 64 (uint64 length)"

    if array.ndim == 1:
        return np.apply_along_axis(
            unpack_bits_fast, 1, array.astype(np.uint)[np.newaxis].T, n=n
        )
    else:
        if array.ndim > 2 or array.shape[1] != 1:
            raise ValueError(
                "Only single row or column vectors are accepted, got: {}".format(array.shape)
            )

        return np.apply_along_axis(unpack_bits_fast, 1, n=n)


def find_ramp_edges(
    trace: np.ndarray,
    *,
    threshold: float = 0.5,
    roll: int = 5,
    roll_fun: callable = np.median,
    kernel: object = None,
    method: str = "local",
    _debug: bool = False,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Gets indexes of falling and rising ramps in a sample trace using convolution with
       an edge detection kernel

    Args:
        trace (np.ndarray): The 1-d array to work on
        threshold (float, optional): Multiplier for lower and upper detection limits
        roll (int, optional): Number of samples to perform rolling preprocessing on
        roll_fun (callable, optional): The preprocessing function to apply
        kernel (object, optional): The ramp-detection kernel
        method (str, optional): Method, either "local" or "gradient"
        _debug (bool, optional): Output additional data for debugging purposes?

    Returns:
        t.Tuple[np.ndarray, np.ndarray]: The falling and rising ramp edges
    """
    assert isinstance(trace, (np.ndarray, pandas.Series))
    assert (trace.ndim == 1) or (trace.ndim == 2 and trace.shape[1] == 1)
    assert isinstance(roll, (int, np.integer))
    assert is_scalar_numeric(threshold) and (threshold > 0.1) and (threshold <= 1.0)
    assert callable(roll_fun)
    assert isinstance(kernel, (type(None), np.ndarray, pandas.Series))

    if kernel is not None:
        if not isinstance(kernel, (np.ndarray, pandas.Series)):
            raise TypeError("'kernel' should be an object usable by np.convolve")
    else:
        kernel = pandas.Series(scipy.signal.windows.hann(16 + 1)).diff().dropna().to_numpy()

    trace = (
        pandas.Series(trace)
        .rolling(roll, min_periods=1, center=True)
        .apply(roll_fun, raw=False)
    )
    convolved = np.convolve(trace, kernel, mode="same")
    convolved[slice(None, kernel.size // 2)] = 0.0
    convolved[slice(-kernel.size // 2, None)] = 0.0

    minmax = np.array([np.nanmin(convolved), np.nanmax(convolved)])
    lower_limit, upper_limit = threshold * minmax
    falling_candidates, *_ = np.where(np.less_equal(convolved, lower_limit))
    rising_candidates, *_ = np.where(np.greater_equal(convolved, upper_limit))

    if method == "local":
        local_minima_values = [
            np.nanmin(convolved[x]) for x in split_into_contiguous_ranges(falling_candidates)
        ]
        local_maxima_values = [
            np.nanmax(convolved[x]) for x in split_into_contiguous_ranges(rising_candidates)
        ]
        local_minima = np.where(np.isin(convolved, local_minima_values))
        local_maxima = np.where(np.isin(convolved, local_maxima_values))

        falling = falling_candidates[np.isin(falling_candidates, local_minima)]
        rising = rising_candidates[np.isin(rising_candidates, local_maxima)]
    else:
        gradient = np.gradient(convolved)
        zero_crossings, *_ = np.where(np.diff(np.sign(gradient)))

        falling = falling_candidates[np.isin(falling_candidates, zero_crossings)]
        rising = rising_candidates[np.isin(rising_candidates, zero_crossings)]

    if not _debug:
        return falling, rising
    else:
        if method == "local":
            return (
                falling,
                rising,
                falling_candidates,
                rising_candidates,
                convolved,
                local_minima,
                local_maxima,
            )
        else:
            return (
                falling,
                rising,
                falling_candidates,
                rising_candidates,
                convolved,
                gradient,
                zero_crossings,
            )


def split_into_contiguous_ranges(trace: np.ndarray, *, step_threshold: int = 1) -> [np.array]:
    """Splits an array into ranges of similar values

    Args:
        trace (np.ndarray): The array to split
        step_threshold (int, optional): The splitting threshold. Defaults to 1.

    Returns:
        [np.array]: The array of split ranges
    """
    return np.split(trace, np.where(np.diff(trace) > step_threshold)[0] + 1)


def get_fft(
    data: pandas.DataFrame,
    *,
    timescale: t.Optional[float] = None,
    demean: bool = True,
    melt: bool = True,
) -> pandas.DataFrame:
    """Gets the fft(s) of data in a timestamps & values DataFrame

    Args:
        data (pandas.DataFrame): The data, including timestamps & values
        timescale (t.Optional[float], optional): If provided, rescales the timing column.
            Otherwise, timestamps in seconds are assumed.
        demean (bool, optional): Subtract the mean from the values?
        melt (bool, optional): Output a melted data frame?

    Returns:
        pandas.DataFrame: The data frame, with FFT values for a range of frequencies. The
            DataFrame holds the original columns, and a frequency column. Optionally, this can be
            melted, with the frequency as the id variable, and values 'melted' into a 'variable'
            column.
    """
    fft = scipy.fft

    # demean
    y = data.iloc[:, 1:] - (data.iloc[:, 1:].mean() if demean else 0)

    # sampling period & frequency
    T = np.float64(data.iloc[:, 0].diff().mean()) * (timescale if timescale else 1.0)
    Fs = np.float64(1.0) / T
    n = data.shape[0]
    r = n // 2

    w = scipy.signal.blackman(n)

    Y = y.apply(lambda x: x * w, 0).apply(fft.fft, 0)
    Y = 2.0 / n * Y.iloc[:r, :]
    xY = fft.fftfreq(n, d=1 / Fs)[:r]
    Y["frequency:fft::Hz"] = xY

    return Y.melt(id_vars="frequency:fft::Hz") if melt else Y


def get_welch(
    data: pandas.DataFrame,
    window_size: t.Optional[int] = None,
    window: str = "blackmanharris",
    *,
    timescale: t.Optional[float] = None,
    demean: bool = True,
    melt: bool = True,
) -> pandas.DataFrame:
    """Gets the fft(s) of data in a timestamps & values DataFrame using the Welch method

    Args:
        data (pandas.DataFrame): The data, including timestamps & values
        window_size (t.Optional[int], optional): The window size
        window (str, optional): The window name
        timescale (t.Optional[float], optional): If provided, rescales the timing column.
            Otherwise, timestamps in seconds are assumed.
        demean (bool, optional): Subtract the mean from the values?
        melt (bool, optional): Output a melted data frame?

    Returns:
        pandas.DataFrame: The data frame, with FFT values for a range of frequencies. The
            DataFrame holds the original columns, and a frequency column. Optionally, this can be
            melted, with the frequency as the id variable, and values 'melted' into a 'variable'
            column.
    """
    y = data.iloc[:, 1:] - (data.iloc[:, 1:].mean() if demean else 0)

    # sampling period & frequency
    T = np.float64(data.iloc[:, 0].diff().mean()) * (timescale if timescale else 1.0)
    Fs = np.float64(1.0) / T
    n = data.shape[0]
    r = n // 2

    _window_size = np.int(window_size) if window_size else n // 16
    _nperseg = _window_size
    _noverlap = _nperseg // 2
    _window = scipy.signal.get_window(window, _window_size)

    # get frequencies first, before applying to columns
    f = scipy.signal.welch(
        y.iloc[:, 0], fs=Fs, window=_window, noverlap=_noverlap, nperseg=_nperseg
    )[0]
    Y = y.apply(
        lambda x: scipy.signal.welch(
            x, fs=Fs, window=_window, noverlap=_noverlap, nperseg=_nperseg
        )[1],
        0,
    )
    Y["frequency:fft::Hz"] = f

    return Y.melt(id_vars="frequency:fft::Hz") if melt else Y
