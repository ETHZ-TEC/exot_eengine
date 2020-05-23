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
"""Huffman source coding"""

import typing as t

import numpy as np
from bitarray import bitarray

from exot.exceptions import *

from ._base import SourceCoding


class Huffman(SourceCoding):
    def __init__(self, *args, **kwargs):
        if "length" not in kwargs:
            raise LayerTypeError("Huffman not provided the 'length' keyword argument")

        code = kwargs.pop("length")

        if not isinstance(code, int):
            raise LayerTypeError("Huffman code 'length' invalid", type(code), int)

        implemented = [4, 5, 8, 9, 16, 18]
        if code not in implemented:
            raise LayerValueError(
                f"Huffman code length {code} not in implemented ({implemented})"
            )

        if code == 4:
            self._codebook = {
                0: bitarray("00"),
                1: bitarray("01"),
                2: bitarray("10"),
                3: bitarray("11"),
            }
        elif code == 5:
            self._codebook = {
                0: bitarray("00"),
                1: bitarray("01"),
                2: bitarray("100"),
                3: bitarray("101"),
                4: bitarray("11"),
            }
        elif code == 8:
            self._codebook = {
                0: bitarray("000"),
                1: bitarray("001"),
                2: bitarray("010"),
                3: bitarray("011"),
                4: bitarray("100"),
                5: bitarray("101"),
                6: bitarray("110"),
                7: bitarray("111"),
            }
        elif code == 9:
            self._codebook = {
                0: bitarray("010"),
                1: bitarray("011"),
                2: bitarray("000"),
                3: bitarray("001"),
                4: bitarray("100"),
                5: bitarray("101"),
                6: bitarray("110"),
                7: bitarray("1110"),
                8: bitarray("1111"),
            }
        elif code == 16:
            self._codebook = {
                0: bitarray("0000"),
                1: bitarray("0001"),
                2: bitarray("0010"),
                3: bitarray("0011"),
                4: bitarray("0100"),
                5: bitarray("0101"),
                6: bitarray("0110"),
                7: bitarray("0111"),
                8: bitarray("1000"),
                9: bitarray("1001"),
                10: bitarray("1010"),
                11: bitarray("1011"),
                12: bitarray("1100"),
                13: bitarray("1101"),
                14: bitarray("1110"),
                15: bitarray("1111"),
            }
        elif code == 18:
            self._codebook = {
                0: bitarray("0100"),
                1: bitarray("0101"),
                2: bitarray("0110"),
                3: bitarray("0111"),
                4: bitarray("0010"),
                5: bitarray("0011"),
                6: bitarray("0000"),
                7: bitarray("0001"),
                8: bitarray("1000"),
                9: bitarray("1001"),
                10: bitarray("1010"),
                11: bitarray("1011"),
                12: bitarray("1100"),
                13: bitarray("1101"),
                14: bitarray("11100"),
                15: bitarray("11101"),
                16: bitarray("11110"),
                17: bitarray("11111"),
            }

        super().__init__(*args, **kwargs)

    def _encode(self, bitstream: t.Union[bitarray, np.ndarray]) -> np.ndarray:
        if isinstance(bitstream, np.ndarray):
            bitstream = bitarray(bitstream.tolist())

        if self.is_uniform:
            # fill to compression factor
            bitstream.extend(1 for _ in range(len(bitstream) % int(self.compression_factor)))
        else:
            _ = bitarray()
            _.encode(self.codebook, np.array(bitstream.decode(self.codebook)))

            for v in self.codebook.values():
                end = bitstream[len(_) :]
                if end in v[0 : len(end)]:
                    break
                bitstream.extend(bit for bit in v[(len(end) - 1) :])

        return np.array(bitstream.decode(self.codebook))

    def _decode(self, symstream: np.ndarray) -> np.ndarray:
        _ = bitarray()
        _.encode(self.codebook, symstream.tolist())
        return np.array(_).astype(int)
