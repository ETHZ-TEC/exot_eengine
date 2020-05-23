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

class EdgeLineCoding(SimpleN):
    def _encode(self, upper):
        return upper

    def __init__(
        self,
        *args,
        threshold: int = 2,
        **kwargs,
    ):
        self.threshold= threshold
        super().__init__(*args, **kwargs)

    def _decode(self, lnestream: np.ndarray) -> np.ndarray:
        assert lnestream.ndim == 2, "only 2-d arrays of symbols can be decoded!"

        ideal = self.config.symstream

        _symbol_space = np.hstack([0, np.diff(np.median(lnestream, axis=1))])
        predictions = np.full(_symbol_space.shape, np.nan)
        predictions[_symbol_space < -self.threshold] = 0
        predictions[_symbol_space >  self.threshold] = 1
        predictions[0] = ideal[0]

        for idx in np.argwhere(np.isnan(predictions)):
            predictions[idx] = predictions[idx-1]

        return predictions

