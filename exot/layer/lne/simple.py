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
"""A simple line-coding layer"""

import typing as t

import numpy as np
import sklearn.base
import sklearn.naive_bayes

from exot.exceptions import *
from exot.util.misc import is_scalar_numeric
from exot.util.scinum import is_fitted

from .._base import Layer


class SimpleN(Layer, layer=Layer.Type.Line):
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
        decision_device: t.Type = sklearn.naive_bayes.GaussianNB,
        **kwargs,
    ):
        self.N_factor = N
        self.default_decision_device = decision_device()
        super().__init__(*args, **kwargs)

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

        self._decision_device = value

    def is_compatible(self, num_symbols: int) -> bool:
        return self.max_num_symbols >= num_symbols

    def _encode_as_symbols(self, symstream: np.ndarray) -> np.ndarray:
        assert (symstream <= self.max_num_symbols - 1).all()
        return symstream

    def _encode(self, symstream: np.ndarray) -> np.ndarray:
        return self._encode_as_symbols(symstream).flatten()

    def _decode(self, lnestream: np.ndarray) -> np.ndarray:
        assert lnestream.ndim == 2, "only 2-d arrays of symbols can be decoded!"

        ideal = self.config.symstream

        lnestream = lnestream[slice(None, len(ideal)), :]
        ideal = ideal[slice(None, len(lnestream))]

        if "decision_device" in self.config:
            self.decision_device = self.config.decision_device
        else:
            self.decision_device = self.default_decision_device

        if not is_fitted(self.decision_device):
            self.decision_device.fit(lnestream, ideal)

        self.add_intermediate("symbol_space", lnestream)
        self.add_intermediate(
            "decision_device",
            pandas.DataFrame({"decision_device": [copy(self.decision_device)]}),
        )

        return self.decision_device.predict(lnestream)
