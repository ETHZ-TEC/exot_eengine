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
"""Error factory class"""

import typing as t

__all__ = ("ErrorFactory",)

"""
ErrorFactory
------------

A helper class for generating a hierarchy of exceptions. Can be used to provide
meaningful errors in classes, while still allowing for
"""


class ErrorFactory:
    def __init__(self, root: str = "", sep: str = "->"):
        assert isinstance(root, str), "'root' has to be a str"
        assert isinstance(sep, str), "'sep' has to be a str"
        self._root = root
        self._sep = sep
        self._prefix = f"{self._root}{self._sep}" if self._root else ""
        self._root_exception = type(f"{self._prefix}Exception", (BaseException,), {})

        common_errors = [
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            NotImplementedError,
        ]
        self._common = {k: self.__call__(k.__name__, k) for k in common_errors}

    @property
    def basic(self) -> t.Tuple:
        """Produce basic error types (1: Value, 2: Type)

        Unpack to produce symbols, for example:

        >>> ValueE, TypeE = ErrorFactory("SomeClass").basic
        """
        return self.common[ValueError], self.common[TypeError]

    @property
    def common(self):
        return self._common

    def __call__(self, name: str = None, *types):
        if not name:
            return self._root_exception
        assert isinstance(name, str)
        assert all(isinstance(t, type) and issubclass(t, BaseException) for t in types)
        _type = type(f"{self._prefix}{name}", (self._root_exception, *types), {})

        if len(types) == 1 and types[0] is TypeError:

            def __str__(self):
                if len(self.args) >= 2:
                    if isinstance(self.args[0], str) and all(
                        isinstance(_, type) for _ in self.args[1:]
                    ):
                        msg, got, *expected = self.args
                        if expected:
                            return "{}, got: {}, expected: one of {!r}".format(
                                msg, got, expected
                            )
                        else:
                            return "{}, got: {}".format(msg, got)
                    elif all(isinstance(_, type) for _ in self.args):
                        got, *expected = self.args
                        return "got: {}, expected: one of {!r}".format(got, expected)
                    else:
                        return str(self.args)
                else:
                    return str(self.args)

            _type.__str__ = __str__

        return _type
