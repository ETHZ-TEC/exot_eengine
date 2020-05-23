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
