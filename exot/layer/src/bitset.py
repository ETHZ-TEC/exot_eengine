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
