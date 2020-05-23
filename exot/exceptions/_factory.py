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
