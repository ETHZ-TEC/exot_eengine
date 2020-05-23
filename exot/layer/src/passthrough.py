"""Passthrough source coding"""

import typing as t

import numpy as np
from bitarray import bitarray

from ._base import SourceCoding


class SourcePassthrough(SourceCoding):
    @property
    def _encode_types(self) -> t.Tuple[type]:
        return (bitarray, np.ndarray)

    @property
    def _decode_types(self) -> t.Tuple[type]:
        return (np.ndarray,)

    @property
    def _encode_validators(self):
        return {np.ndarray: [lambda v: np.isin(v, [0, 1]).all()]}

    def __init__(self, *args, **kwargs):
        self._codebook = {1: bitarray("1"), 0: bitarray("0")}

    def _encode(self, bitstream):
        return np.array(bitstream, dtype=np.dtype(int))

    def _decode(self, symstream):
        return symstream.astype(np.bool).astype(np.dtype(int))

    @property
    def codebook(self):
        return self._codebook
