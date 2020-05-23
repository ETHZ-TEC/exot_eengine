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
"""Helpers to calculate channel capacities."""

import numpy as np

__all__ = ("get_fspectrum", "waterfilling", "capacity_from_connection_matrix")

"""
Signatures
----------
classic_waterfilling            :: (p0, Sqq, Shh) -> Capacity float
constrained_waterfilling        :: (p0, Sqq, Shh) -> Capacity float
capacity_from_connection_matrix :: (A, T_min)     -> Capacity float
"""


def classic_waterfilling(p0, Sqq, Shh):
    """Returns the capacity bound of a given channel determined using classic waterfiling

    See Thermal-CC paper TODO{REF}.

    Args:
        p0: Power cap for the waterfilling algorithm as float
        Sqq: Noise power spectrum as np.darray shape(N,2) where N is the number of frequency
            bins, column 0 holds the frequencies and column 1 the power spectral density.
        Shh: Channel power spectrum as np.darray shape(N,2) where N is the number of frequency
            bins, column 0 holds the frequencies and column 1 the power spectral density.

    Returns:
        Channel Capaicty: in bits per seconds
    """
    _lambda = 1  # Waterfilling parameter
    _alpha = 1  # Lagrangian parameter
    error = np.inf  # Error for input power allocation
    Sxx = np.full(Sqq[:, 0].shape, np.nan)  # Ideal input power allocation
    f_diff = np.concatenate([np.diff(Shh[:, 0]).reshape((-1,)), np.zeros((1,))])

    # Calculate the waterfilling parameter _lambda and consequently the ideal input power allocation Sxx
    while (abs(error) > 10 ** -6) and (_alpha < 10 ** 3):
        p = (1 / _lambda) - (Sqq[:, 1] / Shh[:, 1])
        Sxx[p > 0] = p[p > 0]
        Sxx[p < 0] = 0
        error = (f_diff * Sxx).sum() - p0
        if error > 0:
            _lambda = _lambda * (1 + abs(error) / _alpha)
        else:
            _lambda = _lambda / (1 + abs(error) / _alpha)
        _alpha += 0.01

    return np.log2(1 + ((Sxx * Shh[:, 1]) / Sqq[:, 1])).sum() * np.diff(Sqq[:, 0]).mean()


def constrained_waterfilling(p0, Sqq, Shh):
    """Returns the capacity bound of a given channel determined using constrained waterfiling

    See Thermal-CC paper TODO{REF}.

    Parameters:
        p0: Power cap for the waterfilling algorithm as float
        Sqq: Noise power spectrum as np.darray shape(N,2) where N is the number of frequency
            bins, column 0 holds the frequencies and column 1 the power spectral density.
        Shh: Channel power spectrum as np.darray shape(N,2) where N is the number of frequency
            bins, column 0 holds the frequencies and column 1 the power spectral density.
    Returns:
        Channel Capacity: in bits per seconds
    """
    max_error = 10 ** (-4)  # 9)
    max_alpha = 10 ** (10)

    # Apply withening filter
    N0 = Sqq[:, 1].mean()
    whitening_filter = Sqq[:, 1] / N0
    Sqq_white = Sqq[:, 1] / whitening_filter
    Shh_white = Shh[:, 1] / whitening_filter
    N0_white = Sqq_white.mean()

    # Calculate the capacity C per subband
    C = np.full(N0.shape, np.nan)
    error = np.inf
    df = np.diff(np.concatenate([Shh[:, 0].reshape((-1,)), (Shh[-1, 0]).reshape((-1,))]))
    _lambda = 1
    _alpha = 1
    A_lambda = np.full(Shh_white.shape, True, dtype=bool)
    # Calculate the waterfilling parameter _lambda and consequently the ideal input power allocation Sxx for a subband
    while (abs(error) > max_error) and (_alpha < max_alpha):
        _roh = (1 / 2) * (df[A_lambda] * (_lambda - (1 / Shh_white[A_lambda]))).sum()
        error = _roh - (p0 / N0_white)
        if error > 0:
            _lambda = _lambda / (1 + abs(error) / _alpha)
        else:
            _lambda = _lambda * (1 + abs(error) / _alpha)
        _alpha = _alpha + 0.01
        A_lambda = (Shh_white * _lambda) >= 1
    if abs(error) > max_error:
        print(
            "WARNING: The capacity could only be calculated with an error of " + str(abs(error))
        )
    return (1 / 2) * (df[A_lambda] * np.log2(_lambda * Shh_white[A_lambda])).sum()


def capacity_from_connection_matrix(A, T_min):
    """Returns the capacity bound of a given channel determined for noise free channel based on
    the connection matrix, see Power-CC paper TODO{REF}.

    Parameters:
        A: Transition matrix of the channel model
        T_min: Minimal channel access time in seconds
    Returns:
        Channel Capacity in bits per seconds
        Channel Capacity in bits per channel use
    """
    w = np.linalg.eigvals(A)
    _lambda = w.max()
    return np.log2(_lambda) / T_min, np.log2(_lambda)
