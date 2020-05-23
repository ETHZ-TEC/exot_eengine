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

