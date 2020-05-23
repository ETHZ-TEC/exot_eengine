"""Source coding base class"""

import math
import typing as t

import numpy as np
from bitarray import bitarray

from exot.util.misc import is_scalar_numeric

from .._base import Layer


"""
SourceCoding
------------

The SourceCoding layer accepts the following types:
    - Encoding input:  bitarray, np.ndarray of only bool's or 1's and 0's
    - Encoding output: bitarray, np.ndarray
    - Decoding input:  np.ndarray
    - Decoding output: bitarray, np.ndarray
"""


class SourceCoding(Layer, layer=Layer.Type.Source):
    @property
    def _encode_types(self) -> t.Tuple[type]:
        return (bitarray, np.ndarray)

    @property
    def _decode_types(self) -> t.Tuple[type]:
        return (np.ndarray,)

    @property
    def _encode_validators(self) -> t.Mapping[type, t.Callable[[t.Any], bool]]:
        return {np.ndarray: [lambda v: v.dtype == np.dtype("bool") or np.isin(v, [0, 1]).all()]}

    def symrate_to_bitrate(self, symrate: t.Union[float, int]):
        assert is_scalar_numeric(symrate), "symrate must be float or int"
        return symrate * self.compression_factor

    def bitrate_to_symrate(self, bitrate: t.Union[float, int]):
        assert is_scalar_numeric(bitrate), "bitrate must be float or int"
        return bitrate / self.compression_factor

    def num_syms_to_num_bits(self, num_syms: t.Union[float, int]):
        assert is_scalar_numeric(num_syms), "num_syms must be float or int"
        return math.ceil(num_syms * self.compression_factor)

    def num_bits_to_num_syms(self, num_bits: t.Union[float, int]):
        assert is_scalar_numeric(num_bits), "num_bits must be float or int"
        return math.ceil(num_bits / self.compression_factor)

    @property
    def compression_factor(self) -> float:
        # division with '/' always produces a float
        return sum(len(_) for _ in self.codebook.values()) / len(self.codebook)

    @property
    def is_uniform(self) -> bool:
        return self.compression_factor.is_integer()

    def is_compatible(self, max_num_symbols: int) -> bool:
        return len(self.codebook) <= max_num_symbols

    @property
    def symbols(self):
        return list(self.codebook.keys())

    @property
    def codebook(self) -> dict:
        return self._codebook
