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
"""Symobls are encoded in the median value of the symbol duration"""
import copy as cp
import typing as t

import numpy as np
import pandas as pd
import scipy.interpolate
import sklearn.base
import sklearn.naive_bayes

from exot.exceptions import LayerMisconfigured
from exot.util.scinum import is_fitted

from .simple import SimpleN

class MedianLineCoding(SimpleN):
    def _encode(self, upper):
        return upper

    def create_symbol_space(self, data: np.ndarray) -> np.ndarray:
        """Create a symbol space representation of input data

        Args:
            data (np.ndarray): a 2-d array

        Returns:
            np.ndarray: a symbol space or shape: data.shape[0], phase_count
        """
        _symbol_space = np.zeros((data.shape[0], 1))

        _symbol_space = np.median(data, axis=1)

        return _symbol_space.reshape((-1,1))

    @property
    def symbol_space(self) -> t.Optional[np.ndarray]:
        return getattr(self, "_symbol_space", None)

    def _decode(self, lnestream: np.ndarray) -> np.ndarray:
        assert lnestream.ndim == 2, "only 2-d arrays of symbols can be decoded!"

        ideal = self.config.symstream

        self._symbol_space = self.create_symbol_space(lnestream)
        ideal = self.config.symstream

        self._symbol_space = self._symbol_space[slice(None, len(ideal))]
        ideal = ideal[slice(None, len(self._symbol_space))]

        self.add_intermediate("symbol_space", self._symbol_space)

        if "decision_device" in self.config:
            self.decision_device = self.config.decision_device
        else:
            self.decision_device = sklearn.naive_bayes.GaussianNB()

        if not is_fitted(self.decision_device):
            self.decision_device.fit(self.symbol_space, ideal)

        predictions = self.decision_device.predict(self.symbol_space)
        self.add_intermediate(
            "decision_device",
            pd.DataFrame({"decision_device": [cp.copy(self.decision_device)]}),
        )

        return predictions

