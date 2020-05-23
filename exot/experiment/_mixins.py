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
"""Experiment mixin classes"""

import abc
import enum

import numpy as np
from bitarray import bitarray

__all__ = (
"StreamHandler",
"Ibitstream",
"Isymstream",
"Ilnestream",
"Irdpstream",
"Irawstream",
"Obitstream",
"Osymstream",
"Olnestream",
"Ordpstream",
"Orawstream",
"Oschedules"
)


"""
Input & Output layer properties
===============================

Overview
--------

Layers are composed of `decode`/`encode` pairs, that ideally should form an identity,
such that `encode(decode(x)) == x` and `decode(encode(x)) == x`.


                        decode(...) <--|--> encode(...)
                                       |                                  (schedules)
Output:    o_bitstream -> o_symstream  -> o_lnestream  -> o_rdpstream  -> o_rawstream
                       |               |               |               |
                       |               |               |               |  (measurements)
Input:     i_bitstream <- i_symstream  <- i_lnestream  <- i_rdpstream  <- i_rawstream
                       |               |               |               |
                       |               |               |               |
Layer:                 |-SRC           |-LNE           |-RDP           |-IO
"""
@enum.unique
class StreamType(enum.IntEnum):
    """Stream types"""
    Input  = enum.auto()
    Output = enum.auto()

class StreamHandler(metaclass=abc.ABCMeta):
    @property
    def i_streams(self):
        streams = {}
        for cls in type(self).__bases__:
            if hasattr(cls, 'is_stream'):
                if(cls.is_stream()[0] == StreamType.Input):
                    streams[cls.is_stream()[1]] = getattr(self, cls.is_stream()[2])
        return streams

    @property
    def o_streams(self):
        streams = {}
        for cls in type(self).__bases__:
            if hasattr(cls, 'is_stream'):
                if(cls.is_stream()[0] == StreamType.Output):
                    streams[cls.is_stream()[1]] = getattr(self, cls.is_stream()[2])
        return streams

class Obitstream(metaclass=abc.ABCMeta):
    # 1. bitstream -- raw bits, a bitarray
    @staticmethod
    def is_stream():
        return (StreamType.Output, "bitstream", "o_bitstream")

    @property
    def o_bitstream(self):
        return getattr(self, "_o_bitstream", None)

    @o_bitstream.setter
    def o_bitstream(self, value):
        assert isinstance(value, (bitarray, np.ndarray))
        setattr(self, "_o_bitstream", value)

    @o_bitstream.deleter
    def o_bitstream(self):
        if hasattr(self, "_o_bitstream"):
            delattr(self, "_o_bitstream")

class Osymstream(metaclass=abc.ABCMeta):
    # 2. symstream -- symbols, produced by the src layer
    @staticmethod
    def is_stream():
        return (StreamType.Output, "symstream", "o_symstream")

    @property
    def o_symstream(self):
        return getattr(self, "_o_symstream", None)

    @o_symstream.setter
    def o_symstream(self, value):
        setattr(self, "_o_symstream", value)

    @o_symstream.deleter
    def o_symstream(self):
        if hasattr(self, "_o_symstream"):
            delattr(self, "_o_symstream")

class Olnestream(metaclass=abc.ABCMeta):
    # 3. lnestream -- line-encoded symbols, produced by the lne layer
    @staticmethod
    def is_stream():
        return (StreamType.Output, "lnestream", "o_lnestream")

    @property
    def o_lnestream(self):
        return getattr(self, "_o_lnestream", None)

    @o_lnestream.setter
    def o_lnestream(self, value):
        setattr(self, "_o_lnestream", value)

    @o_lnestream.deleter
    def o_lnestream(self):
        if hasattr(self, "_o_lnestream"):
            delattr(self, "_o_lnestream")

class Ordpstream(metaclass=abc.ABCMeta):
    # 4. rdpstream -- post-processed symbols, produced by the rdp layer
    @staticmethod
    def is_stream():
        return (StreamType.Output, "rdpstream", "o_rdpstream")

    @property
    def o_rdpstream(self):
        return getattr(self, "_o_rdpstream", None)

    @o_rdpstream.setter
    def o_rdpstream(self, value):
        setattr(self, "_o_rdpstream", value)

    @o_rdpstream.deleter
    def o_rdpstream(self):
        if hasattr(self, "_o_rdpstream"):
            delattr(self, "_o_rdpstream")

class Orawstream(metaclass=abc.ABCMeta):
    # 5. rawstream -- outstream adjusted for the source app, produced by the io layer
    @staticmethod
    def is_stream():
        return (StreamType.Output, "rawstream", "o_rawstream")

    @property
    def o_rawstream(self):
        return getattr(self, "_o_rawstream", None)

    @o_rawstream.setter
    def o_rawstream(self, value):
        setattr(self, "_o_rawstream", value)

    @o_rawstream.deleter
    def o_rawstream(self):
        if hasattr(self, "_o_rawstream"):
            delattr(self, "_o_rawstream")

class Oschedules(metaclass=abc.ABCMeta):
    # 5. schedules -- individual schedules for the source app, produced by the io layer
    @property
    def o_schedules(self):
        return getattr(self, "_o_schedules", None)

    @o_schedules.setter
    def o_schedules(self, value):
        setattr(self, "_o_schedules", value)

    @o_schedules.deleter
    def o_schedules(self):
        if hasattr(self, "_o_schedules"):
            delattr(self, "_o_schedules")


class Irawstream(metaclass=abc.ABCMeta):
    # 1. rawstream -- raw input from the sink app, ingested by the io layer
    @staticmethod
    def is_stream():
        return (StreamType.Input, "rawstream", "i_rawstream")

    @property
    def i_rawstream(self):
        return getattr(self, "_i_rawstream", None)

    @i_rawstream.setter
    def i_rawstream(self, value):
        setattr(self, "_i_rawstream", value)

    @i_rawstream.deleter
    def i_rawstream(self):
        if hasattr(self, "_i_rawstream"):
            delattr(self, "_i_rawstream")

class Irdpstream(metaclass=abc.ABCMeta):
    # 2. rdpstream -- input to the rdp module, produced by the io layer
    @staticmethod
    def is_stream():
        return (StreamType.Input, "rdpstream", "i_rdpstream")

    @property
    def i_rdpstream(self):
        return getattr(self, "_i_rdpstream", None)

    @i_rdpstream.setter
    def i_rdpstream(self, value):
        setattr(self, "_i_rdpstream", value)

    @i_rdpstream.deleter
    def i_rdpstream(self):
        if hasattr(self, "_i_rdpstream"):
            delattr(self, "_i_rdpstream")

class Ilnestream(metaclass=abc.ABCMeta):
    # 2. lnestream -- input to the lne module, produced by the rdp layer
    @staticmethod
    def is_stream():
        return (StreamType.Input, "lnestream", "i_lnestream")

    @property
    def i_lnestream(self):
        return getattr(self, "_i_lnestream", None)

    @i_lnestream.setter
    def i_lnestream(self, value):
        setattr(self, "_i_lnestream", value)

    @i_lnestream.deleter
    def i_lnestream(self):
        if hasattr(self, "_i_lnestream"):
            delattr(self, "_i_lnestream")

class Isymstream(metaclass=abc.ABCMeta):
    # 2. symstream -- input to the src module, produced by the lne layer
    @staticmethod
    def is_stream():
        return (StreamType.Input, "symstream", "i_symstream")

    @property
    def i_symstream(self):
        return getattr(self, "_i_symstream", None)

    @i_symstream.setter
    def i_symstream(self, value):
        setattr(self, "_i_symstream", value)

    @i_symstream.deleter
    def i_symstream(self):
        if hasattr(self, "_i_symstream"):
            delattr(self, "_i_symstream")

class Ibitstream(metaclass=abc.ABCMeta):
    # 2. bitstream -- final output bits, produced by the src layer
    @staticmethod
    def is_stream():
        return (StreamType.Input, "bitstream", "i_bitstream")

    @property
    def i_bitstream(self):
        return getattr(self, "_i_bitstream", None)

    @i_bitstream.setter
    def i_bitstream(self, value):
        if isinstance(
            value, (bitarray, np.ndarray)
        ):  # f"value is of type {type(value)}, should be bitarray or np.ndarray"
            setattr(self, "_i_bitstream", value)

    @i_bitstream.deleter
    def i_bitstream(self):
        if hasattr(self, "_i_bitstream"):
            delattr(self, "_i_bitstream")

