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
"""Custom decorators"""

import typing as t
from functools import partial, wraps

__all__ = (
    "only_once",
    "dummy",
    "verbose",
    "noexcept",
    "decorate_methods",
    "read_only",
    "make_read_only",
)


def only_once(f: t.Callable[..., t.T]) -> t.Callable[..., t.T]:
    """Decorator for running a function only once

    Args:
        f (t.Callable[..., t.T]): The function to be decorated

    Returns:
        t.Callable[..., t.T]: A decorated function
    """

    @wraps(f)
    def wrapper(*_args, **_kwargs) -> t.Optional[t.T]:
        if not wrapper.__once:
            wrapper._once = True
            return f(*_args, **_kwargs)

    wrapper.__once: bool = False
    return wrapper


def dummy(f: t.Callable) -> t.Callable:
    """Dummy forwarding decorator

    Args:
        f (t.Callable): The function to be decorated

    Returns:
        t.Callable: a decorated function
    """

    @wraps(f)
    def wrapper(*_args, **_kwargs):
        return f(*_args, **_kwargs)

    return wrapper


def verbose(f: t.Callable) -> t.Callable:
    """Decorator for printing function arguments before a call

    Args:
        f (t.Callable): The function to be decorated

    Returns:
        t.Callable: A decorated function
    """
    from sys import stderr

    @wraps(f)
    def wrapper(*_args, **_kwargs):
        if hasattr(f, "__self__") and getattr(f, "__self__", None):
            wrapper._name = f"{f.__self__.__class__.__name__}.{f.__name__}"
        else:
            wrapper._name = f.__name__
        print(f"{wrapper._name} called with args: {_args}, kwargs: {_kwargs}", file=stderr)
        return f(*_args, **_kwargs)

    return wrapper


def noexcept(f: t.Callable) -> t.Callable:
    """Decorator for ignoring exceptions

    Args:
        f (t.Callable): The function to be decorated

    Returns:
        t.Callable: A decorated function
    """
    from sys import stderr

    @wraps(f)
    def wrapper(*_args, **_kwargs):
        try:
            result = f(*_args, **_kwargs)
        except Exception as e:
            print(f"{f.__name__} would throw: {e!r}", file=stderr)
        return result

    return wrapper


def decorate_methods(
    decorator: t.Callable[..., t.Callable[..., t.Callable]] = dummy,
    methods: t.Optional[t.List[str]] = None,
) -> t.Callable:
    """Class/object decorator that decorates its methods

    Args:
        decorator (t.Callable[..., t.Callable[..., t.Callable]]): The decorator function.
            Optional. Defaults to 'dummy'.
        methods (t.Optional[t.List[str]]): The list of methods to decorate. Defaults to None.

    Returns:
        t.Callable: A decorator

    Raises:
        TypeError: Wrong type provided to 'decorator'

    The function can also be used to produce partially-evaluated versions of itself,
    e.g. to create a `make_methods_verbose` decorator.

    To produce partially-evaluated decorator helpers call this function on an
    empty (None) argument. For example:

    >>> decorate_methods(decorator=None, methods=['print'])()

    will return a functools.partial object that is partially evaluated w.r.t.
    'methods'. Similarly:

    >>> decorate_methods(decorator=verbose)()

    will return a functools.partial object that will apply the {verbose} decorator
    to function methods. It can later be used as:
    ::

        # Make a new decorator
        make_methods_verbose = decorate_methods(decorator=verbose)()
        # Apply the decorator to a class definition
        @make_methods_verbose(methods=["__setitem__", "__getitem__"])
        class MyClass(dict): pass
        class SomeClass(dict): pass
        # Decorate an existing class
        ModifiedClass = make_methods_verbose(methods=[ "__getitem__"])(SomeClass)
        # Decorate an object
        make_methods_verbose(methods=[ "__setitem__"])(SomeClass()).__setitem__(1, 1)
    """

    def wrapper(klass: t.T = None) -> t.Union[t.T, partial]:
        """
        :param klass: type/object to decorate, defaults to None
        :type klass: t.T, optional
        :returns: a decorated type/object, or a partially evaluated `decorate_methods`
        :rtype: {t.Union[t.T, partial]}
        :raises: TypeError
        """
        if klass is None:
            spec = (decorator is None, methods is None)

            if spec == (False, False):
                return partial(decorate_methods, decorator=decorator, methods=methods)
            elif spec == (False, True):
                return partial(decorate_methods, decorator=decorator)
            elif spec == (True, False):
                return partial(decorate_methods, methods=methods)
            else:
                return decorate_methods

        if not issubclass(type(decorator), t.Callable):
            raise TypeError("'decorator' must be a t.Callable type")

        if isinstance(klass, type):
            _ = klass.__name__

            class klass(klass):
                pass

            klass.__name__ = _

        available_methods = [
            m
            for m in type(klass).__dir__(klass)
            if issubclass(type(getattr(klass, m, None)), t.Callable)
            and m not in ["__class__", "__new__"]
        ]

        if methods is not None:
            if not issubclass(type(methods), t.List):
                raise TypeError("'methods' needs to be a list")
            to_decorate = filter(lambda x: x in methods, available_methods)
        else:
            to_decorate = available_methods

        for method in to_decorate:
            m = getattr(klass, method, None)
            if m is not None:
                decorated = decorator(m)
                setattr(klass, method, decorated)

        return klass

    return wrapper


class read_only:
    """Read only property using the descriptor protocol

    ..note::
        The read_only properties/attributes work only on instances, and will not
        work on standalone values. Attempting:

        >>> x = read_only(123.45)
        >>> x = "Something else"

        will simply reassing 'x'. However, it works when used as follows:

        >>> class X:
                x = read_only(123.45)
        >>> Instance = X()
        >>> print(Instance.x) # accesses correctly
        >>> Instance.x = "Something else" # fails!
    """

    def __init__(self, value):
        self.value = value

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        raise AttributeError("read-only attribute")


def make_read_only(*attributes: str) -> t.Callable:
    """A class decorator that makes certain attributes read-only

    Args:
        *attributes (str): A list of attributes to make read-only

    Returns:
        t.Callable: A wrapper function

    Raises:
        ValueError: If any of  *attributes is not a `str`
    """

    def dummy(klass):
        return klass

    if not attributes:
        return dummy

    if any(not isinstance(x, str) for x in attributes):
        raise ValueError("'attributes' should contain only strings", attributes)

    def wrapper(klass: type = None) -> t.Union[type, partial]:
        if klass is None:
            return partial(make_read_only, attributes=attributes)
        if not isinstance(klass, type):
            raise TypeError("decorator can only be applied to classes", klass)

        class klass(klass):
            pass

        for attribute in attributes:
            if hasattr(klass, attribute):
                setattr(klass, attribute, read_only(getattr(klass, attribute)))

        return klass

    return wrapper
