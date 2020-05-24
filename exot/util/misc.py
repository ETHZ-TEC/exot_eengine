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
"""Misc helpers"""

import math
import random
import re
import signal
import typing as t
from datetime import datetime
from enum import Enum
from functools import reduce
from inspect import isabstract
from string import ascii_letters
from subprocess import list2cmdline as _list2cmdline
from typing import Mapping as Map

import numpy as np

from exot.exceptions import *

__all__ = (
    "call_with_leaves",
    "dict_depth",
    "dict_diff",
    "find_attributes",
    "flatten_dict",
    "get_concrete_subclasses",
    "get_subclasses",
    "get_valid_access_paths",
    "getitem",
    "has_method",
    "has_property",
    "has_type",
    "has_variable",
    "is_abstract",
    "is_scalar_numeric",
    "leaves",
    "list2cmdline",
    "map_to_leaves",
    "mro_getattr",
    "mro_hasattr",
    "random_string",
    "safe_eval",
    "sanitise_ansi",
    "setgetattr",
    "setitem",
    "stub_recursively",
    "unpack__all__",
    "validate_helper",
    "get_cores_and_schedules",
)


"""
Signatures
----------
call_with_leaves        :: (function: Callable[[Any], Any], obj: ~T, _seq: bool = True) -> None
dict_depth              :: (obj: Any, level: int = 0) -> int
dict_diff               :: (left: Mapping, right: Mapping) -> List[Dict]
find_attributes         :: (attr: str, klass: Any) -> List
flatten_dict            :: (obj: Mapping, sep: str = '.') -> Mapping
get_concrete_subclasses :: (klass, recursive=True, derived=True) -> List
get_subclasses          :: (klass, recursive=True, derived=True) -> List
get_valid_access_paths  :: (obj: Mapping, _limit: int = 8192, _leaf_only: bool = False, _use_lists: bool = True, _fallthrough_empty: bool = True) -> Generator
getitem                 :: (obj: Mapping, query: Union[str, Tuple], *args: Any, sep: str = '/') -> Any
has_method              :: (klass: Union[type, object], name: str) -> bool
has_property            :: (klass: Union[type, object], name: str) -> bool
has_type                :: (klass: Union[type, object]) -> bool
has_variable            :: (klass: Union[type, object], name: str) -> bool
is_abstract             :: (klass: Union[type, object]) -> bool
is_scalar_numeric       :: (value: t.Any) -> bool
map_to_leaves           :: (function: Callable[[Any], Any], obj: ~T, _seq: bool = True) -> Any
mro_getattr             :: (cls: type, attr: str, *args: Any) -> Any
mro_hasattr             :: (cls: type, attr: str) -> bool
random_string           :: (length: int) -> str
safe_eval               :: (to_eval: str, expect: Tuple[type], timeout: int) -> object
sanitise_ansi           :: (value Union[List[str], str]) -> Union[List[str], str]
setgetattr              :: (klass: Union[type, object], attr: str, default: Any) -> None
setitem                 :: (obj: MutableMapping, query: Tuple, value: Any) -> None
stub_recursively        :: (obj: ~T, stub: Any = None, _stub_list_elements: bool = True) -> Optional[~T]
unpack__all__           :: (*imports: Collection[str]) -> Tuple[str]
validate_helper         :: (what: Mapping, key: Any, *types: type, msg: str = '') -> NoReturn
"""


def call_with_leaves(function: t.Callable[[t.Any], t.Any], obj: t.T, _seq: bool = True) -> None:
    """Calls a function on leaves of an object

    A leaf is considered to be an object that is not a Mapping (or, when _seq is set,
    also not a Sequence except a string, which is also a Sequence).

    Args:
        function (t.Callable[[t.Any], t.Any]): The callable
        obj (t.T): The tree-like or sequence-like object
        _seq (bool, optional): Should sequences be considered?. Defaults to True.
    """

    def inner(obj: t.T) -> t.Any:
        if isinstance(obj, Map):
            for v in obj.values():
                inner(v)
        elif _seq and isinstance(obj, (t.List, t.Set)):
            for v in obj:
                inner(v)
        else:
            return function(obj)

    inner(obj)


def dict_depth(obj: t.Any, level: int = 0) -> int:
    """Get maximum depth of a dict-like object

    Args:
        obj (t.Any): The dict-like object
        level (int): For internal use only. Defaults to 0.

    .. note::
        The depth of a non-dict-like object is considered to be 0.
        An empty dict increases the depth if `_empty_increments` is True.

        Examples:

        >>> dict_depth(1)                         # returns 0
        >>> dict_depth([1,2,3])                   # returns 0
        >>> dict_depth({1: 1, 2: 2})              # returns 1
        >>> dict_depth({1: {2: {3: 3}}})          # returns 3
        >>> dict_depth({1: {2: {3: {}}}})         # returns 4
    """
    if not isinstance(obj, Map) or not obj:
        return level
    return max(dict_depth(v, level + 1) for k, v in obj.items())


def dict_diff(left: Map, right: Map) -> t.List[t.Dict]:
    """Get the difference between 2 dict-like objects

    Args:
        left (Map): The left dict-like object
        right (Map): The right dict-like object

    The value returned is a list of dictionaries with keys ["path", "left", "right"]
    which contain the query path and the differences between the left and right mapping.
    If a key is missing in either mapping, it will be indicated as a "None".

    `math.nan` (not-a-number) is used for default values in the comparison because of
    the property: `math.nan != math.nan`. Simple None cannot be used, since it would
    not handle keys that both have a value of None. In general, this function might
    report false-positives for keys that contain the math.nan (or np.nan) value simply
    due to this property. There is no workaround available.
    """
    left_paths = set(get_valid_access_paths(left, _leaf_only=True, _use_lists=False))
    right_paths = set(get_valid_access_paths(right, _leaf_only=True, _use_lists=False))

    return list(
        {
            "path": path,
            "left": getitem(left, path, math.nan),
            "right": getitem(right, path, math.nan),
        }
        for path in left_paths.union(right_paths)
        if getitem(left, path, math.nan) != getitem(right, path, math.nan)
    )


def find_attributes(klass: t.Any, attr: str) -> t.List:
    """Find attributes in any of a class'es bases

    Args:
        klass (t.Any): The type object
        attr (str): The attribute

    Returns:
        t.List: List of found instances of the attribute in the class hierarchy
    """
    if not isinstance(attr, str):
        raise TypeError(attr)

    mro = klass.__mro__ if hasattr(klass, "__mro__") else type(klass).mro()
    return [attr for base in mro if hasattr(base, attr)]


def flatten_dict(obj: Map, sep: str = ".") -> Map:
    """Flatten a dict to a 1-level dict combining keys with a separator

    Args:
        obj (Map): The dict-like object
        sep (str): The separator used when combining keys. Defaults to ".".

    Returns:
        Map: A flattened object of same type as 'obj'.

    .. warning::
        Flattening will enforce all keys to be string-types!

    `reducer` is a function accepted by the functools.reduce function, which is of
    form: f(a, b) where _a_ is the accumulated value, and _b_ is the updated value
    from the iterable.

    The .items() function produces key-value tuple-pairs. These can be expanded
    with *, e.g. `*("a", "b")` will expand to `"a", "b"`. This property is used
    to expand the `kv_pair` below.

    Example walkthrough on `flatten_dict({'a': 1, 'b': {'c': {'d': 2}}})`: ::

        `outer` <- obj: {'a': 1, 'b': {'c': {'d': 2}}}, prefix: ''
            `reducer` <- key: 'a', value: 1
                `inner` <- acc: {}, key: 'a', value: 1, prefix: ''
                `inner` -> {'a': 1}
            `reducer` -> {'a': 1}
            `reducer` <- key: 'b', value: {'c': {'d': 2}}
                `inner` <- acc: {'a': 1}, key: 'b', value: {'c': {'d': 2}}, prefix: ''
        `outer` <- obj: {'c': {'d': 2}}, prefix: 'b.'
            `reducer` <- key: 'c', value: {'d': 2}
                `inner` <- acc: {}, key: 'c', value: {'d': 2}, prefix: 'b.'
        `outer` <- obj: {'d': 2}, prefix: 'b.c.'
            `reducer` <- key: 'd', value: 2
                `inner` <- acc: {}, key: 'd', value: 2, prefix: 'b.c.'
                `inner` -> {'b.c.d': 2}
            `reducer` -> {'b.c.d': 2}
        `outer` -> {'b.c.d': 2}
                `inner` -> {'b.c.d': 2}
            `reducer` -> {'b.c.d': 2}
        `outer` -> {'b.c.d': 2}
                `inner` -> {'a': 1, 'b.c.d': 2}
            `reducer` -> {'a': 1, 'b.c.d': 2}
        `outer` -> {'a': 1, 'b.c.d': 2}
    """
    if not isinstance(obj, Map):
        raise TypeError("flatten_dict works only on dict-like types", type(obj))

    _t = type(obj)

    def outer(obj: Map, prefix: str) -> Map:
        def reducer(accumulator: Map, kv_pair: t.Tuple):
            return inner(accumulator, *kv_pair, prefix)

        return reduce(reducer, obj.items(), _t())

    def inner(accumulator: Map, key: str, value: t.Any, prefix: str) -> Map:
        if isinstance(value, Map):
            return _t(**accumulator, **outer(value, prefix + key + sep))
        else:
            return _t(**accumulator, **_t({prefix + key: value}))

    return outer(obj, "")


def expand_dict(obj: Map, sep: str = ".") -> Map:
    """Expands a flattened mapping by splitting keys with the given separator

    Args:
        obj (Map): The flattened dict-like object to unflatten
        sep (str, optional): The key separator

    Raises:
        TypeError: If wrong type is supplied
        ValueError: If a non-flat dict is supplied

    Returns:
        Map: The expanded mapping object of same type as 'obj'.

    Example:
        >>> d = {'a': 1, 'b': 2, 'c.ca': 1, 'c.cb': 2}
        >>> expand_dict(d)
        {'a': 1, 'b': 2, 'c': {'ca': 1, 'cb': 2}}
    """
    if not isinstance(obj, Map):
        raise TypeError("expand_dict works only on dict-like types", type(obj))
    if dict_depth(obj) != 1:
        raise ValueError(
            "expand_dict works only on flat dict-like types, "
            "got a mapping of depth: {}".format(dict_depth(obj))
        )

    def inner(obj):
        accumulator = type(obj)()

        for k, v in obj.items():
            *head, last = k.split(sep)
            _ = accumulator

            # Create missing paths
            for part in head:
                if part not in _:
                    _[part] = type(obj)()
                _ = _[part]

            _[last] = v

        return accumulator

    return inner(obj)


def get_concrete_subclasses(klass, recursive: bool = True, derived: bool = True) -> t.List:
    """Get a list of non-abstract subclasses of a type

    Args:
        klass (t.Type): The type object
        recursive (bool): Should the classes be extracted recursively? Defaults to True.
        derived (bool): Use the 'derived' property of SubclassTracker-enhanced types? [True]

    Returns:
        t.List: A list of concrete subclasses of the type
    """
    from exot.util.mixins import _SubclassTracker as __

    if derived and hasattr(klass, __.concrete):
        return list(getattr(klass, __.concrete))

    subclasses = get_subclasses(klass, recursive=recursive)
    return [k for k in subclasses if not isabstract(k)]


def get_subclasses(klass, recursive: bool = True, derived: bool = True) -> t.List:
    """Get a list of subclasses of a type

    Args:
        klass (t.Type): The type object
        recursive (bool): Should the classes be extracted recursively? Defaults to True.
        derived (bool): Use the 'derived' property of SubclassTracker-enhanced types? [True]

    Returns:
        t.List: A list of concrete subclasses of the type
    """
    from exot.util.mixins import _SubclassTracker as __

    if not (hasattr(klass, "__subclasses__") or hasattr(klass, __.derived)):
        raise TypeError(f"__subclasses__ or {__.derived} attribute missing", klass)

    if derived:
        return list(getattr(klass, __.derived))

    subclasses = klass.__subclasses__()

    def walker(k):
        first, *rest = k

        if len(rest):
            walker(rest)

        if first not in subclasses:
            subclasses.append(first)

        if hasattr(first, "__subclasses__"):
            _ = first.__subclasses__()
            if len(_):
                walker(_)

    if recursive:
        walker(subclasses)

    return subclasses


def get_valid_access_paths(
    obj: Map,
    _limit: int = 8192,
    _leaf_only: bool = False,
    _use_lists: bool = True,
    _fallthrough_empty: bool = True,
) -> t.Generator[t.Tuple, None, None]:
    """Generate valid key sequences in a dict, optionally including lists

    Args:
        obj (Map): The dict-like object
        _limit (int): Maximum number of paths that can be created with list-like elements.
        _leaf_only (bool): Provide paths for only the leaves of the mapping. Defaults to True.
        _use_lists (bool): Provide paths for list-like elements in the mapping. Defaults to True.
        _fallthrough_empty (bool): Discard empty list- or dict-like elements? Defaults to True.

    Details:
        If `_leaf_only` is set, only paths to leaves will be produced, a leaf being a value
        that is not a mapping (or list).
        If `_use_lists` is set, lists will also be *recursively* checked for valid paths.
        if `_fallthrough_empty` is set, an empty dict or list will yield an empty tuple,
        rendering a parent path.

    Returns:
        t.Generator[t.Tuple,None,None]: A generator that yields the access paths (tuples).

    Examples:
    >>> # Only leaves:
    >>> d = {'a1': {'a2': None}, 'b2': None}
    >>> list(get_valid_access_paths(d, _leaf_only=True))
    [('a1', 'a2'), ('b2',)]
    >>> # All paths:
    >>> list(get_valid_access_paths(d, _leaf_only=False))
    [('a1',), ('a1', 'a2'), ('b2',)]
    """

    def thrower(o: object, t: type, n: str) -> t.NoReturn:
        if not isinstance(o, t):
            raise TypeError(
                f"get_valid_access_paths expected {t!r} for {n!r}, got: {type(o)!r}"
            )

    thrower(obj, Map, "obj")
    thrower(_limit, int, "_limit")
    thrower(_leaf_only, bool, "_leaf_only")
    thrower(_use_lists, bool, "_use_lists")
    thrower(_fallthrough_empty, bool, "_fallthrough_empty")

    def inner(obj: t.Union[Map, t.List, t.Set]) -> t.Generator:
        if _fallthrough_empty and not obj:
            yield tuple()

        # if obj is a mapping
        if isinstance(obj, Map):
            for k, v in obj.items():
                # if the value in obj is also a mapping...
                if isinstance(v, Map):
                    if not _leaf_only:
                        yield (k,)

                    # ... make a recursive call
                    for vv in inner(v):
                        yield (k,) + vv

                # if the value in obj is a list...
                elif _use_lists and isinstance(v, (t.List, t.Set)):
                    # ... first yield the valid path to the key containing the list
                    if v and not _leaf_only:
                        yield (k,)
                    elif not v and _fallthrough_empty:
                        yield (k,)

                    # ... loop through elements, and keep track of indexes
                    for idx, vv in enumerate(v):

                        # if an element is also a mapping or list...
                        if isinstance(vv, (Map, (t.List, t.Set))):
                            # ... make a recursive call
                            for vvv in inner(vv):
                                yield (k,) + (idx,) + vvv
                        else:
                            # ... otherwise yield keypath + idx
                            yield (k,) + (idx,)

                # if the value is neither a mapping nor a list, yield the key
                else:
                    yield (k,)

        # if obj is a list-like sequence
        if _use_lists and isinstance(obj, (t.List, t.Set)):
            # might be tricky to generate valid sequences for large lists!
            if _limit and len(obj) >= _limit:
                raise ValueError(
                    f"get_key_sequences list limit of {_limit} exceeded: {len(obj)}"
                )

            for idx, v in enumerate(obj):
                if isinstance(v, (Map, (t.List, t.Set))):
                    for vv in inner(v):
                        yield (idx,) + vv
                else:
                    yield (idx,)

    return inner(obj)


def getitem(obj: Map, query: t.Union[str, t.Tuple], *args: t.Any, sep: str = "/") -> t.Any:
    """Get a value from a dict-like object using an XPath-like query, or a tuple-path

    Accesses an object that provides a dict-like interface using a query: either a
    tuple representing the path, or a string where consecutive keys are separated with
    a separator, e.g. "key1/key2".

    Returns the value of the object at the given key-sequence. Returns a default value
    if provided, or throws a LookupError.

    Args:
        obj (Map): a mapping
        query (t.Union[str, t.Tuple]): a query path using a separated string or a tuple
        *args (t.Any): an optional default value, similar to `getattr`
        sep (str, optional): a separator string used to split a string query path

    Returns:
        t.Any: the value stored in obj for the given query, or the default value

    Raises:
        LookupError: if query not found and no default value is provided
        TypeError: if obj is not a mapping, or query is not a str or tuple
    """

    if not isinstance(obj, Map):
        raise TypeError("'obj' must be an instance of Mapping, e.g. dict", type(obj))
    if not isinstance(query, (str, t.Tuple)):
        raise TypeError("'query' must be a str or a tuple", type(query))

    if len(args) > 1:
        raise TypeError(f"getitem accepts at most 3 positional args, got {len(args)}")

    _obj = obj

    # handler for tuple queries
    if isinstance(query, t.Tuple):
        _valid = get_valid_access_paths(obj)

        if query not in _valid:
            if args:
                return args[0]
            else:
                raise LookupError(f"query {query!r} not found")
        else:
            for node in query:
                _obj = _obj[node]

        return _obj

    # handler for string queries
    else:
        try:
            # loop through components in the query, consecutively accessing the mapping
            for node in query.split(sep):
                # handle empty nodes in the query, e.g. when query="a///b" -> "a/b"
                if not node:
                    continue

                if isinstance(_obj, Map):
                    for k in _obj.keys():
                        node = type(k)(node) if str(k) == node else node
                elif isinstance(_obj, (t.List, t.Set)):
                    try:
                        node = int(node)
                    except TypeError:
                        raise LookupError(
                            f"{node} not convertible to int when attempting to access "
                            f"a list {_obj!r}"
                        )

                _obj = _obj[node]

            return _obj
        except LookupError as Error:
            if args:
                return args[0]
            else:
                Error.args += (query,)
                raise


def has_method(klass: t.Union[type, object], name: str) -> bool:
    """Check if a method exists in any of a klass'es bases

    Args:
        klass (t.Union[type, object]): The type or object
        name (str): The name of the method

    Returns:
        bool: True if has a method with the given name.
    """

    candidates = find_attributes(klass, name)
    if not candidates:
        return False

    def is_callable(c):
        return isinstance(getattr(klass, str(c), None), t.Callable)

    return all(is_callable(f) for f in candidates)


def has_property(klass: t.Union[type, object], name: str) -> bool:
    """Check if a variable exists in any of a klass'es bases

    Args:
        klass (t.Union[type, object]): The type or object
        name (str): The name of the property

    Returns:
        bool: True if has a property with the given name.
    """

    candidates = find_attributes(klass, name)
    if not candidates:
        return False

    def is_property(c):
        return not isinstance(getattr(klass, str(c), None), property)

    return all(is_property(f) for f in candidates)


def has_type(klass: t.Union[type, object]) -> bool:
    """Check if a type or instance has a Type member type that derives from Enum

    Args:
        klass (t.Union[type, object]): The type or object

    Returns:
        bool: True if has the "Type" attribute.
    """
    if not isinstance(klass, (type, object)):
        raise TypeError(klass)

    return issubclass(getattr(klass, "Type", type(None)), Enum)


def has_variable(klass: t.Union[type, object], name: str) -> bool:
    """Check if a variable exists in any of a klass'es bases

    Args:
        klass (t.Union[type, object]): The type or object
        name (str): The name of the variable

    Returns:
        bool: True if has a variable with the given name.
    """

    candidates = find_attributes(klass, name)
    if not candidates:
        return False

    def is_not_callable(c):
        return not isinstance(getattr(klass, str(c), None), t.Callable)

    return all(is_not_callable(f) for f in candidates)


def is_abstract(klass: t.Union[type, object]) -> bool:
    """Check if a type or instance is abstract

    Args:
        klass (t.Union[type, object]): The type or object

    Returns:
        bool: True if the type/instance is abstract.
    """
    if not isinstance(klass, (type, object)):
        raise TypeError(klass)

    if hasattr(klass, "__abstractmethods__"):
        return 0 != len(getattr(klass, "__abstractmethods__"))
    else:
        from inspect import isabstract

        return isabstract(klass)


def is_scalar_numeric(value: t.Any) -> bool:
    """Check if is an int, a float, or a NumPy variant thereof

    Args:
        value (t.Any): The value to inspect

    Returns:
        bool: True if scalar and numeric.
    """
    return isinstance(value, (float, int, np.integer, np.floating))


def leaves(obj: Map) -> t.Generator:
    """Get leaves of a mapping

    Args:
        obj (Map): The dict-like object

    Returns:
        t.Generator: A generator that yields the leaf elements of the mapping.
    """
    paths = get_valid_access_paths(obj, _leaf_only=True, _use_lists=False)
    return (getitem(obj, path) for path in paths)


def list2cmdline(seq: t.Iterable) -> str:
    """Translates a sequence of arguments into a command line string with "None" removal

    Args:
        seq (t.Iterable): The sequence of arguments

    Returns:
        str: The command-line string
    """

    seq = [_ for _ in seq if _ is not None]
    return _list2cmdline(seq)


def map_to_leaves(function: t.Callable[[t.Any], t.Any], obj: t.T, _seq: bool = True) -> t.Any:
    """Map a function to leaves of an object

    A leaf is considered to be an object that is not a Mapping (or, when _seq is set,
    also not a Sequence except a string, which is also a Sequence).

    Args:
        function (t.Callable[[t.Any], t.Any]): a function or signatude "a -> a"
        obj (t.T): a dict-like, list-like, or plain object
        _seq (bool, optional): map on elements of lists?

    Returns:
        t.T: the obj with transformed elements
    """

    def inner(obj: t.T) -> t.Any:
        if isinstance(obj, Map):
            return type(obj)({k: inner(v) for k, v in obj.items()})
        elif _seq and isinstance(obj, (t.List, t.Set)):
            return type(obj)(inner(v) for v in obj)
        else:
            return function(obj)

    return inner(obj)


def mro_getattr(cls: type, attr: str, *args: t.Any) -> t.Any:
    """Get an attribute from a type's class hierarchy

    Args:
        cls (type): The type
        attr (str): The attribute
        *args (t.Any): The default value (like in Python's default getattr)

    Returns:
        t.Any: The attribute, or when not found the default value (if provided)

    Raises:
        TypeError: Not called on a type
        TypeError: Wrong number of arguments
        AttributeError: Attribute not found and no default value provided
    """

    if not isinstance(cls, type):
        raise TypeError(f"mro_getattr can only be used on types, got {type(cls)}")

    if len(args) > 1:
        raise TypeError(f"mro_getattr expected at most 3 arguments, got {2 + len(args)}")

    for klass in cls.mro()[1:]:
        if hasattr(klass, attr):
            # return first matching attribute
            return getattr(klass, attr)

    if args:
        # if provided, return args[0], i.e. the a default value
        return args[0]
    else:
        raise AttributeError(f"type object {cls.__name__!r} has not attribute {attr!r}")


def mro_hasattr(cls: type, attr: str) -> bool:
    """Check if an attribute exists in a type's class hierarchy

    Args:
        cls (type): The type
        attr (str): The attribute

    Returns:
        bool: True if has the attribute.

    Raises:
        TypeError: Not called on a type
    """

    if not isinstance(cls, type):
        raise TypeError(f"mro_getattr can only be used on types, got {type(cls)}")

    for klass in cls.mro()[1:]:
        if hasattr(klass, attr):
            return True
    return False


def random_string(length: int) -> str:
    """Make a random string of specified length

    Args:
        length (int): The desired random string length

    Returns:
        str: The random string
    """
    assert isinstance(length, int), f"'length' must be an int, got: {type(length)}"
    return "".join(random.choices(ascii_letters, k=length))


def timestamp() -> str:
    """Make a timestamp with current time

    Returns:
        str: The timestamp in ISO format
    """
    return datetime.now().isoformat("_", timespec="seconds").replace(":", "-")


def safe_eval(
    to_eval: str, *, expect: t.Tuple[type] = (list, np.ndarray), timeout: int = 10
) -> object:
    """Evaluate a restricted subset of Python (and numpy) from a string

    Args:
        to_eval (str): The string to evaluate
        expect (t.Tuple[type]): The list of expected resulting types. Defaults to list, ndarray.
        timeout (int): The timeout after which the call fails in seconds. Defaults to 10.

    The `safe_eval` function allows using a subset of commands, listed in `_globals` and
    `_locals`, which includes a few numpy functions: linspace, arange, array, rand, and
    randint. Examples:

    >>> safe_eval("linspace(1, 10, 10, dtype=int).tolist()")
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> safe_eval("__import__('os').getcwd()")
    NameError  Traceback (most recent call last)
    ...
    NameError: name '__import__' is not defined
    >>> safe_eval("range(5)")
    TypeError  Traceback (most recent call last)
    ...
    TypeError: eval produced a <class 'range'>, expected: (<class 'list'>, <class 'numpy.ndarray'>)
    >>> safe_eval("list(round(rand(), 2) for _ in range(5))")
    [0.96, 0.41, 0.9, 0.98, 0.02]
    """
    assert isinstance(to_eval, str), "'to_eval' must be a str"
    assert isinstance(expect, tuple), "'expect' must be a tuple"
    assert all(isinstance(_, type) for _ in expect), "'expect' must contain only types"

    _locals = {}
    _globals = {
        "__builtins__": {},
        "list": list,
        "range": range,
        "len": len,
        "int": int,
        "float": float,
        "min": min,
        "max": max,
        "round": round,
        "linspace": np.linspace,
        "geomspace": np.geomspace,
        "logspace": np.logspace,
        "hstack": np.hstack,
        "vstack": np.vstack,
        "split": np.split,
        "arange": np.arange,
        "array": np.array,
        "rand": np.random.rand,
        "randint": np.random.randint,
    }

    class AlarmException(Exception):
        pass

    def signal_handler(number: int, frame):
        assert number == signal.SIGALRM.value
        raise AlarmException()

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(timeout)

    try:
        _ = eval(to_eval, _globals, _locals)
    except AlarmException:
        raise TimeoutError(f"safe_eval took longer than {timeout} seconds")
    else:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        signal.alarm(0)

    if not isinstance(_, expect):
        raise EvalTypeError(f"eval produced a {type(_)}, expected: {expect}")

    return _


def sanitise_ansi(value: t.Union[t.List[str], str]) -> t.Union[t.List[str], str]:
    """Remove all ANSI escape characters from a str or a list of str

    Args:
        value (t.Union[t.List[str], str]): The string or list of strings

    Returns:
        t.Union[t.List[str], str]: The sanitised string or a list of sanitised strings
    """
    _ansi_escape = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")

    if isinstance(value, str):
        return _ansi_escape.sub("", value)
    elif isinstance(value, t.List):
        return list(map(lambda x: _ansi_escape.sub("", x).strip(), value))
    else:
        raise TypeError("sanitise_ansi accepts only str or lists of str")


def setgetattr(klass: t.Union[type, object], attr: str, default: t.Any) -> None:
    """Combines `setattr` and `getattr` to set attributes

    Args:
        klass (t.Union[type, object]): The type or object
        attr (str): The attribute
        default (t.Any): The default value
    """
    if not any([isinstance(klass, type), isinstance(klass, object)]):
        raise TypeError("'klass' should be a type or an object", klass)
    if not isinstance(attr, str):
        raise TypeError("'attr' should be a str")
    if not attr:
        raise ValueError("'attr' should not be empty")

    setattr(klass, attr, getattr(klass, attr, default))


def setitem(obj: t.MutableMapping, query: t.Tuple, value: t.Any, force: bool = False) -> None:
    """Set a value in a dict-like object using a tuple-path query

    Args:
        obj (t.MutableMapping): a mutable mapping
        query (t.Tuple): a query path as a tuple
        value (t.Any): value to set

    Raises:
        TypeError: if obj is not a mutable mapping
    """

    if not isinstance(obj, t.MutableMapping):
        raise TypeError("'obj' needs to be a mutable mapping", type(obj))

    _obj = obj
    _valid = get_valid_access_paths(obj)

    if query not in _valid:
        if not force:
            raise KeyError(f"query-path {query!r} not found")
        else:
            for node in query[:-1]:
                if node not in _obj:
                    _obj = dict()
                _obj = _obj[node]
    else:
        for node in query[:-1]:
            _obj = _obj[node]

    _obj[query[-1]] = value


def stub_recursively(
    obj: t.T, stub: t.Any = None, _stub_list_elements: bool = True
) -> t.Optional[t.T]:
    """Produce a copy with all leaf values recursively set to a 'stub' value

    Args:
        obj (t.T): the object to stub
        stub (t.Any, optional): the value to set the leaf elements to
        _stub_list_elements (bool, optional): stub individual elements in collections?

    Returns:
        (t.T, optional): the stubbed object
    """

    def inner(obj):
        if isinstance(obj, Map):
            return type(obj)((k, inner(v)) for k, v in obj.items())
        elif _stub_list_elements and isinstance(obj, (t.List, t.Set)):
            return type(obj)(inner(v) for v in obj)
        else:
            return stub

    return inner(obj)


def unpack__all__(*imports: t.Collection[str]) -> t.Tuple[str]:
    """Upacks a list of lists/tuples into a 1-dimensional list

    Args:
        *imports (t.Collection[str]): The collections of strings in "__all__"

    Returns:
        t.Tuple[str]: The flattened imports as a tuple of strings.
    """
    from itertools import chain

    _name = f"{__name__}.unpack__all__"

    if not all(isinstance(e, (t.List, t.Tuple)) for e in imports):
        raise TypeError(f"{_name}: arguments should be lists or tuples")

    _ = chain(*imports)

    assert all(
        issubclass(type(e), str) for e in _
    ), f"{_name}: values in unpacked containers were not scalar or 'str'"

    return tuple(_)


def validate_helper(what: t.Mapping, key: t.Any, *types: type, msg: str = "") -> t.NoReturn:
    """Validate types of key in a mapping using key-paths

    Args:
        what (t.Mapping): The mapping
        key (t.Any): The key
        *types (type): The valid types
        msg (str): An additional error message. Defaults to "".
    """
    if not isinstance(what, t.Mapping):
        raise TypeError(f"validate_helper works only on mappings, got {type(what)}")
    if not types:
        raise TypeError(f"validate helper expects at least 1 'types' argument")

    if isinstance(key, str) or not isinstance(key, t.Iterable):
        key = tuple([key])
    elif not isinstance(key, tuple):
        key = tuple(key)

    # The `config` property setter guarantees that `config` is a fully
    # mutated AttributeDict, therefore :meth:`getattr` can be used.
    if not isinstance(getitem(what, key, None), types):
        raise MisconfiguredError(
            "{0}config key: '{1!s}' should be of type {2!r}, got {3!s}".format(
                f"{msg} " if msg else "", key, types, type(getitem(what, key, None))
            )
        )


def get_cores_and_schedules(environments_apps_zones: t.Mapping) -> set:
    e_a_z = environments_apps_zones
    _cores_and_schedules = set()

    for env in e_a_z:
        for app in e_a_z[env]:
            if app != "src":
                continue

            _path_to_cores = ("app_config", "generator", "cores")
            _path_to_schedule_tag = ("zone_config", "schedule_tag")

            access_paths = list(get_valid_access_paths(e_a_z[env][app]))

            if _path_to_cores not in access_paths:
                raise LayerMisconfigured(
                    f"{env!r}->{app!r} must have a 'generator.cores' config key"
                )
            if _path_to_schedule_tag not in access_paths:
                _ = e_a_z[env][app]["zone"]
                raise LayerMisconfigured(
                    f"{env!r}.{_!r} of app {app!r} must have a schedule_tag"
                )

            _cores_and_schedules.add(
                (
                    len(getitem(e_a_z[env][app], _path_to_cores)),
                    getitem(e_a_z[env][app], _path_to_schedule_tag),
                )
            )

    return _cores_and_schedules
