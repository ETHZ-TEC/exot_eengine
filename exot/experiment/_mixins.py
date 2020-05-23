"""Experiment mixin classes"""

import abc

import numpy as np
from bitarray import bitarray

__all__ = ("CCOStreams", "CCIStreams")


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


class CCOStreams(metaclass=abc.ABCMeta):
    @property
    def o_streams(self):
        return {
            "bitstream": self.o_bitstream,
            "symstream": self.o_symstream,
            "lnestream": self.o_lnestream,
            "rdpstream": self.o_rdpstream,
            "rawstream": self.o_rawstream,
        }

    # 1. bitstream -- raw bits, a bitarray
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

    # 2. symstream -- symbols, produced by the src layer
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

    # 3. lnestream -- line-encoded symbols, produced by the lne layer
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

    # 4. rdpstream -- post-processed symbols, produced by the rdp layer
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

    # 5. rawstream -- outstream adjusted for the source app, produced by the io layer
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


class CCIStreams(metaclass=abc.ABCMeta):
    @property
    def i_streams(self):
        return {
            "bitstream": self.i_bitstream,
            "symstream": self.i_symstream,
            "lnestream": self.i_lnestream,
            "rdpstream": self.i_rdpstream,
            "rawstream": self.i_rawstream,
        }

    # 1. rawstream -- raw input from the sink app, ingested by the io layer
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

    # 2. rdpstream -- input to the rdp module, produced by the io layer
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

    # 2. lnestream -- input to the lne module, produced by the rdp layer
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

    # 2. symstream -- input to the src module, produced by the lne layer
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

    # 2. bitstream -- final output bits, produced by the src layer
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


class SweepOStreams(CCIStreams):
    @property
    def o_streams(self):
        return {
            "lnestream": self.o_lnestream,
            "rdpstream": self.o_rdpstream,
            "rawstream": self.o_rawstream,
        }


class SweepIStreams(CCIStreams):
    @property
    def i_streams(self):
        return {
            "lnestream": self.i_lnestream,
            "rdpstream": self.i_rdpstream,
            "rawstream": self.i_rawstream,
        }


class SweepOStreams(CCOStreams):
    @property
    def o_streams(self):
        return {
            "lnestream": self.o_lnestream,
            "rdpstream": self.o_rdpstream,
            "rawstream": self.o_rawstream,
        }


class ExploratoryIStreams(CCIStreams):
    @property
    def i_streams(self):
        return {"rdpstream": self.i_rdpstream, "rawstream": self.i_rawstream}


class ExploratoryOStreams(CCOStreams):
    @property
    def o_streams(self):
        return {"rdpstream": self.o_rdpstream, "rawstream": self.o_rawstream}
