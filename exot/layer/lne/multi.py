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
"""Line-coding layer for multi-channel encodings"""

import typing as t
from copy import copy

import numpy as np
import pandas
import sklearn.base
import sklearn.naive_bayes
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm

from exot.exceptions import *
from exot.util.misc import is_scalar_numeric
from exot.util.scinum import is_fitted, pack_array, unpack_array

from .._base import Layer


class MultiN(Layer, layer=Layer.Type.Line):
    @property
    def _encode_types(self) -> t.Tuple[type]:
        return (np.ndarray,)

    @property
    def _decode_types(self) -> t.Tuple[type]:
        return (np.ndarray,)

    @property
    def _encode_validators(self) -> t.Mapping[type, t.Callable[[object], bool]]:
        return {np.ndarray: [lambda x: (x <= self.max_num_symbols).all() and (x >= 0).all()]}

    @property
    def _decode_validators(self) -> t.Mapping[type, t.Callable[[object], bool]]:
        return {}

    @property
    def required_config_keys(self) -> t.List[str]:
        return ["symstream"]

    def validate(self) -> t.NoReturn:
        """Implements method `validate` from Configurable

        Raises:
            LayerMisconfigured: If wrong symstream was provided
        """
        _ = self.config.symstream

        if not isinstance(_, np.ndarray):
            raise LayerMisconfigured("symstream not an array", type(_), np.ndarray)

        if not (_.ndim == 1) or (_.ndim == 2 and _.shape[1] == 1):
            raise LayerMisconfigured(f"symstream not trivial to flatten: ({_.shape})")

    @property
    def requires_runtime_config(self) -> bool:
        """Decoding needs access to 'ideal_symstream'"""
        return (False, True)

    def __init__(
        self,
        *args,
        N: int = 64,
        decision_device=sklearn.pipeline.make_pipeline(sklearn.svm.LinearSVC()),
        reducer=None,
        **kwargs,
    ):
        self.N_factor = N
        self.default_decision_device = decision_device
        self.reducer = self.default_reducer if not reducer else reducer

        super().__init__(*args, **kwargs)

    @staticmethod
    def default_reducer(x, axis):
        return np.array([np.median(x, axis=axis)]).T

    @property
    def N_factor(self):
        return self._N_factor

    @N_factor.setter
    def N_factor(self, value):
        if not is_scalar_numeric(value):
            raise LayerMisconfigured("N factor must be a scalar numeric value")

        self._N_factor = int(value)

    @property
    def max_num_symbols(self) -> int:
        return 2 ** self.N_factor

    def symrate_to_subsymrate(self, symrate: t.T) -> t.T:
        return symrate

    @property
    def decision_device(self) -> t.Optional[object]:
        return getattr(self, "_decision_device", None)

    @decision_device.setter
    def decision_device(self, value) -> None:
        if isinstance(value, type):
            raise LayerMisconfigured("'decision_device' must be an object, not a type")
        if not sklearn.base.is_classifier(value):
            raise LayerMisconfigured("'decision_device' can only be a classifier")

        self._decision_device = value

    @property
    def default_decision_device(self) -> t.Optional[object]:
        return getattr(self, "_default_decision_device", None)

    @default_decision_device.setter
    def default_decision_device(self, value) -> None:
        if isinstance(value, type):
            raise LayerMisconfigured("'default_decision_device' must be an object, not a type")
        if not sklearn.base.is_classifier(value):
            raise LayerMisconfigured("'default_decision_device' can only be a classifier")

        self._default_decision_device = value

    def is_compatible(self, num_symbols: int) -> bool:
        return self.max_num_symbols >= num_symbols

    def _encode_as_symbols(self, symstream: np.ndarray) -> np.ndarray:
        assert (symstream <= self.max_num_symbols - 1).all()
        return symstream

    @property
    def reducer(self):
        return getattr(self, "_reducer", None)

    @reducer.setter
    def reducer(self, value):
        assert callable(value), "must be callable"
        self._reducer = value

    def _encode(self, symstream: np.ndarray) -> np.ndarray:
        return self._encode_as_symbols(symstream).flatten()

    def _decode(self, lnestream: np.ndarray) -> np.ndarray:
        assert lnestream.ndim == 3, "only 3-d arrays of symbols can be decoded!"

        # The lnestream produced by the RDP layer is structured as follows:
        #
        #   Axis 0: symbol space (should be of same size as symstream); each plane in axis 0
        #           should represent 1 symbol, which in turn is represented in N individual
        #           bits of axis 1. For example, with 50 symbols in the symbol stream, the
        #           shape of lnestream should be (50, _, _).
        #   Axis 1: bit space (each plane in axis 1 should be a single cache line); for
        #           example, with 4 cache lines the shape of lnestream should be (_, 4, _).
        #   Axis 2: sample space (each row represents a number of samples taken for specific
        #           cache line for each data bit); for example, with 64 samples, the shape
        #           of lnestream should be (_, _, 64).
        #
        # To sum up: for 50 symbols, each represented with 4 cache lines, and 64 samples per
        # each, the shape of lnestream should be (50, 4, 64).
        #
        # Since we adopt an MSB encoding, the lnestream has to be flipped in axis 1 (cache
        # line M represents the Mth bit of the symbol).
        lnestream = np.vstack(np.flip(lnestream, axis=1))
        ideal = unpack_array(self.config.symstream, n=self.N_factor).flatten()

        lnestream = lnestream[slice(None, len(ideal)), :]
        ideal = ideal[slice(None, len(lnestream))]

        # Apply the reducer
        self._reduced_lnestream = np.apply_over_axes(self._reducer, lnestream, 1)

        if "decision_device" in self.config:
            self.decision_device = self.config.decision_device
        else:
            self.decision_device = copy(self.default_decision_device)

        if not is_fitted(self.decision_device):
            self.decision_device.fit(self._reduced_lnestream, ideal)

        predictions = self.decision_device.predict(self._reduced_lnestream)

        self.add_intermediate("symbol_space", self._reduced_lnestream)
        self.add_intermediate(
            "decision_device",
            pandas.DataFrame({"decision_device": [copy(self.decision_device)]}),
        )

        return pack_array(predictions, n=self.N_factor, pad="lsb")
