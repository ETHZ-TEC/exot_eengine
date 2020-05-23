"""Frequency Governor related line codings"""
import abc
import copy as cp
import typing as t

import numpy as np

from exot.exceptions import LayerMisconfigured

from .._base import Layer

class FrequencyGovernorLineCoding(Layer, layer=Layer.Type.Line):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def symrate_to_subsymrate(self, symrate: t.T) -> t.T:
        """Convert a symbol rate to a subsymbol rate"""
        return symrate * self.subsymbol_count

    def _encode(self, upper):
        """
        """
        symbols = np.unique(upper)
        if(symbols.min() < 0 or symbols.max() > 1 or symbols.size > 2):
            raise LayerTypeError(f"Symbols not fitting for layer, only accpts 0 and 1 but got {symbols}")

        lower = np.empty((upper.size,self.subsymbol_count))
        for v in self.codebook:
          lower[upper==v,:] = self.codebook[v]
        lower = lower.flatten()
        lower = np.hstack([self.preamble, lower, self.postamble])

        return lower

    @property
    @abc.abstractmethod
    def subsymbol_count(self):
        pass

    @property
    @abc.abstractmethod
    def codebook(self):
        pass

    @property
    @abc.abstractmethod
    def preamble(self):
        pass


    @property
    @abc.abstractmethod
    def postamble(self):
        pass

class ConservativeGovLineCoding(FrequencyGovernorLineCoding):
    def __init__(
        self,
        *,
        center: int = 10,
        step: int = 2,
        **kwargs,
    ):
        """Initialise the Conservative Governor Line Coding layer

        Args:
        """
        self.center = center
        self.step   = step

        super().__init__(**kwargs)

    @property
    def subsymbol_count(self):
        return 2

    @property
    def codebook(self):
      return {
          0:[self.step, -self.center],
          1:[-self.step, self.center]
          }

    @property
    def preamble(self):
        return self.center

    @property
    def postamble(self):
        return -self.center


    def _decode(self, lnestream: np.ndarray) -> np.ndarray:
        """Decodes the stream with a simple majority voting scheme.

        Args:

        Returns:
            np.ndarray: a n-d array
        """
        raise DeprecationWarning("This layer has never been tested and is only a re-implementation of similar functionality from a previous ExOT version")

        assert lnestream.ndim == 1, "only 1-d arrays of symbols can be decoded!"

        center_idxes = np.where(lnestream==self.center)[0]
        center_idxes = np.hstack([0,center_idxes[np.diff(center_idxes)>1],-1])

        symbols = []
        for idx in range(len(center_idxes)-1):
          zero_count = np.where(lnestream[center_idxes[idx]:center_idxes[idx+1]] == self.center+self.step)[0].size
          one__count = np.where(lnestream[center_idxes[idx]:center_idxes[idx+1]] == self.center-self.step)[0].size

          if(zero_count > 0 and one__count > 0):
              raise Exception("This should not happen...")

          if(zero_count > one_count):
              symbols.append(0)
          else:
              symbols.append(1)

        return np.array(symbols)



