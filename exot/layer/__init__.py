"""Communication-oriented layers"""

from . import _base, _factory, io, lne, rdp, src
from ._base import Layer
from ._factory import LayerFactory

"""
In Python, class definitions are executed like regular code. Due to the lack of
import guards circular dependencies can arise. Base classes have been separated
from `__init__` files to avoid that. Components that use the base `Layer` need
to be imported after the base class.

All objects from used modules are imported in this `__init__` file so that they
are registered with the `SubclassTracker` and usable in the `LayerFactory`.
"""


__all__ = ("Layer", "LayerFactory", "io", "lne", "rdp", "src")

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
