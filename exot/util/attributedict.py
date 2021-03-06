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
"""A dictionary with attribute access"""

from __future__ import annotations

from collections.abc import MutableMapping
from keyword import kwlist
from typing import Any, Callable, Dict, Hashable, Iterable, List, Type, Union
from warnings import warn as _warn

__all__ = ("AttributeDict", "LabelMapping")

ENABLE_WARNINGS = False


def warn(*args, **kwargs):
    if ENABLE_WARNINGS:
        _warn(*args, **kwargs)
    else:
        pass


class AttributeDict(MutableMapping):
    """A dict with attribute access

    In addition to functions defined in the abstract base classes of Mapping and
    MutableMapping, provides the {__getattr__}, {__setattr__}, and {__delattr__}
    functions, for the 'dotted' attribute access.

    AttributeDict methods perform 'forwarding' to an internal dict object which holds
    the data. Attempting to set a key of type 'dict' will cast it to this type, to
    ensure consistent access 'recursively'.

    Extends MutableMapping.

    Attributes:
        _locals (List[str]): local member variables
        _warn_list (List[str]): list of keys which might be tricky to set/access
        _mutate_on_init (bool): flag to recursively mutate contained data
    """

    # the `__slots__` magic attribute prevents a `__dict__` being created on instances
    __slots__ = ["_data"]

    _locals: List[str] = ["_data", "_warn_list", "_locals", "_mutate_on_init", "__slots__"]
    _warn_list: List[str] = sum(
        [
            _locals,
            # Implemented methods
            [
                "__init__",
                "__getattr__",
                "__setattr__",
                "__delattr__",
                "__getstate__",
                "__setstate__",
                "copy",
                "fromkeys",
                "from_dict",
                "to_dict",
                "as_dict",
                "mutate",
            ],
            # Overriden abstract methods
            list(MutableMapping.__abstractmethods__),
            # Methods provided by MutableMapping, including stub abstract ones.
            [k for k, v in MutableMapping.__dict__.items() if isinstance(v, Callable)],
            # Restricted keywords
            kwlist,
        ],
        [],
    )
    _mutate_on_init: bool = True

    def __init__(self, *args: Dict, **kwargs: Any):
        """Initialise an instance"""
        if len(args) > 1:
            raise TypeError(f"at most 1 positional argument is expected, got {len(args)}")
        self._data = {}
        if args:
            self.update(args[0])
        if len(kwargs):
            self.update(kwargs)
        if self._mutate_on_init:
            self.mutate(to=type(self))

    def __len__(self):
        """Get number of entries in the dict

        Implements an abstract method from MutableMapping.
        """
        return len(self._data)

    def __getitem__(self, key: Hashable):
        """Get an item from the dict

        Implements an abstract method from MutableMapping.
        """
        if key in self._data:
            return self._data[key]
        # Refer to built-in dict's docs for usage of the "__missing__" method
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)

    def __setitem__(self, key: Hashable, value: Any):
        """Set/add an item in the dict

        Implements an abstract method from MutableMapping.
        """
        if "." in key or key in self._warn_list:
            _ = f"setting key {key!r} may prevent attribute-based access"
            warn(_, RuntimeWarning)
        self._data[key] = value

    def __delitem__(self, key: Hashable):
        """Delete an item in the dict

        Implements an abstract method from MutableMapping.
        """
        del self._data[key]

    def __iter__(self):
        """Get an iterator for the dict

        Implements an abstract method from MutableMapping.
        """
        return iter(self._data)

    def __contains__(self, key: Hashable):
        """Check if an item is contained in the dict

        Implements an abstract method from MutableMapping.
        """
        return key in self._data

    def __repr__(self):
        """Get a representation of the dict

        Implements an abstract method from MutableMapping.
        """
        return repr(self._data)

    def __getattr__(self, key: Hashable):
        """Get an attribute from the internal dict or elsewhere in the instance
        """
        try:
            # try to get an attribute via an `object` proxy
            return super().__getattribute__(key)
        except AttributeError:
            # if an attribute is not found, try to get an item from the contained dict
            try:
                return self.__getitem__(key)
            except KeyError:
                raise AttributeError(key)

    def __setattr__(self, key: Hashable, value: Any):
        """Set an attribute in the instance or set/add an item in the contained dict
        """
        if key in self._locals:
            # Pass-through for local attributes
            super().__setattr__(key, value)
            return

        try:
            # if an attribute can be found in a regular fashion...
            super().__getattribute__(key)
        except AttributeError:
            # if the attribute does not already exist in the instance, attempt to
            # set/add an item to the contained dict.
            try:
                self.__setitem__(key, value)
            except KeyError:
                raise AttributeError(key)
        else:
            # ... set it on the instance via its `object` proxy
            super().__setattr__(key, value)

    def __delattr__(self, key: Hashable) -> None:
        """Delete an attribute in the instance or an item in the contained dict
        """
        try:
            # if an attribute can be found in a regular fashion...
            super().__getattribute__(key)
        except AttributeError:
            try:
                self.__delitem__(key)
            except KeyError:
                raise AttributeError(key)
        else:
            # ... set it on the instance via its `object` proxy
            super().__delattr__(key)

    def __getstate__(self):
        """Get state for serialisation

        `_data` is the only 'state' that the AttributeDict is holding. Without this
        method a recursion error may arise during serialisation (e.g. to YAML).
        """
        # return {slot: getattr(self, slot) for slot in self.__slots__}
        return {"_data": self.as_dict()}

    def __setstate__(self, value):
        """Set state from a serialised representation"""
        # for slot in value: setattr(self, slot, value[slot])
        setattr(self, "_data", value["_data"])
        if self._mutate_on_init:
            self.mutate(to=type(self))

    def copy(self) -> AttributeDict:
        """Make a copy of the AttributeDict instance

        Implementation of an analogous method of the built-in dict.
        """
        if self.__class__ is AttributeDict:
            assert isinstance(self._data, dict), "self._data should have been a dict"
            return AttributeDict(self._data.copy())
        else:
            import copy

            data = self._data
            try:
                self._data = {}
                copied = copy.copy(self)
            finally:
                self._data = data
            copied.update(self)
            return copied

    @classmethod
    def fromkeys(cls, iterable: Iterable, value: Any = None) -> AttributeDict:
        """Produce an AttributeDict using an iterable type as keys

        Implementation of an analogous method of the built-in dict.

        Args:
            iterable (Iterable): The iterable
            value (Any): Value to which the new keys are set, defaults to None

        Returns:
            AttributeDict: The dict created from the iterable
        """
        o = cls()
        for key in iterable:
            o[key] = value
        return o

    @classmethod
    def from_dict(cls, o: Any) -> AttributeDict:
        """Produce an AttributeDict from a built-in dict

        Args:
            o (Any): A Dict when called, other types when called recursively.

        Returns:
            AttributeDict: The converted dict
        """
        if isinstance(o, MutableMapping):
            return cls((k, cls.from_dict(v)) for k, v in o.items())
        elif isinstance(o, List):
            return type(o)(cls.from_dict(v) for v in o)
        else:
            return o

    @classmethod
    def to_dict(cls, o: Any) -> Dict:
        """Convert an AttributeDict to a built-in dict recursively

        Args:
            o (Any): An AttributeDict when called, other types when called

        Returns:
            Dict: A plain built-in dict
        """
        if isinstance(o, MutableMapping):
            return dict((k, cls.to_dict(v)) for k, v in o.items())
        elif isinstance(o, List):
            return type(o)(cls.to_dict(v) for v in o)
        else:
            return o

    def as_dict(self) -> dict:
        """Get a copy of the instance converted to the built-in dict
        """
        return self.to_dict(self)

    def mutate(self, to: Union[AttributeDict, Dict, Type[AttributeDict], Type[Dict]]) -> None:
        """Mutate the internal data to AttributeDict or dict, recursively

        Args:
            to (Union[AttributeDict, Dict, Type[AttributeDict], Type[Dict]]): The type
                to which the internal data is to be mutated
        """
        if isinstance(to, (AttributeDict, AttributeDict.__class__)):
            self._data = dict((k, AttributeDict.from_dict(v)) for k, v in self._data.items())
        elif issubclass(to, MutableMapping):
            self._data = AttributeDict.to_dict(self)
        else:
            raise TypeError(f"'to' is {type(to)}, should be either AttributeDict or Dict")


class LabelMapping(AttributeDict):
    def __init__(self, *args: Dict, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if "UNKNOWN" not in self._data.keys():
            self.UNKNOWN = {
                "int": max([self._data[x]["int"] for x in self._data]) + 1,
                "str": "-",
            }

    def __missing__(self, key: Hashable):
        return self.UNKNOWN
