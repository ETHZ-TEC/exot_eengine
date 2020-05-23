import typing as t

import numpy as np
from bitarray import bitarray

from exot.exceptions import *
from exot.util.scinum import pack_array, unpack_array

from ._base import SourceCoding


class BitsetCoding(SourceCoding):
    def __init__(self, *args, bitset_length: int, **kwargs):
        self.bitset_length = bitset_length
        super().__init__(*args, **kwargs)

    @property
    def bitset_length(self):
        return self._bitset_length

    @bitset_length.setter
    def bitset_length(self, value: int):
        if not isinstance(value, (int, np.uint)):
            raise LayerMisconfigured(
                "Bitset length must be an integer, got: {!r}".format(type(value))
            )

        if (value > 64) or (value < 1):
            raise LayerMisconfigured(
                "Bitset coding requires length in range [1, 64], got: {}".format(value)
            )

        self._bitset_length = np.uint(value)

    @property
    def compression_factor(self) -> float:
        return float(self.bitset_length)

    @property
    def is_uniform(self) -> bool:
        return True

    def is_compatible(self, max_num_symbols: int) -> bool:
        return (2 ** self.bitset_length) <= max_num_symbols

    @property
    def symbols(self):
        return NotImplemented

    @property
    def codebook(self) -> dict:
        return NotImplemented

    def _encode(self, bitstream: t.Union[bitarray, np.ndarray]) -> np.ndarray:
        if isinstance(bitstream, bitarray):
            bitstream = np.array(bitarray.tolist(), dtype=int)
        return pack_array(bitstream, self.bitset_length, pad="lsb").astype(np.uint64)

    def _decode(self, symstream: np.ndarray) -> bitarray:
        return unpack_array(symstream, self.bitset_length).flatten()
