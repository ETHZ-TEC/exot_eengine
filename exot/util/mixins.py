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
"""Mixin helper classes"""

import abc
import copy
import io
import itertools
import os
import pickle
import typing as t
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from exot.exceptions import *
from exot.util.attributedict import AttributeDict
from exot.util.decorators import read_only
from exot.util.file import check_access
from exot.util.logging import get_root_logger
from exot.util.misc import (
    flatten_dict,
    get_valid_access_paths,
    getitem,
    has_method,
    has_property,
    has_type,
    is_abstract,
    leaves,
    mro_getattr,
    mro_hasattr,
    setgetattr,
    setitem,
    stub_recursively,
)

# Ignore NaturalNameWarning(s) produced by the pytables module
warnings.filterwarnings("ignore", module="tables")


try:
    from yaml import CDumper as Dumper, CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

__all__ = (
    "Configurable",
    "HasParent",
    "SubclassTracker",
    "Serialisable",
    "Pickleable",
    "YamlSerialisable",
    "IntermediatesHolder",
    "IntermediatesHandler",
    "LabelConversions",
)


"""
Configurable
------------

Configurable mixin can be used to add to a class an ability to add a `config` property
that performs type checks and verifies that a provided dict-like structure contains
at least a required set of keys. Moreover,
"""

# shorthands for attribute names
class _Configurable(SimpleNamespace):
    config_arg = read_only("__config_arg__")


class Configurable(metaclass=abc.ABCMeta):
    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        """Initialise a subclass of Configurable"""
        if "configure" in kwargs:
            configure = kwargs.pop("configure")
        else:
            # the default value
            configure = mro_getattr(cls, _Configurable.config_arg, "config")

        if not configure:
            raise ValueError(
                "empty 'configure' in creation of {cls.__name__} subclass", configure, cls
            )

        elif not isinstance(configure, str):
            raise TypeError("'configure' is not a str", configure, cls)

        setattr(cls, _Configurable.config_arg, configure)
        super().__init_subclass__(**kwargs)

    def __init__(self, *args: t.Dict, **kwargs: t.Dict) -> None:
        """Initialise the Configurable"""
        if len(args) > 1:
            raise ValueError("at most 1 positional argument is accepted")

        self._config = None  # type: t.Optional[AttributeDict]

        if args:
            self.config = args[0]
        else:
            config_arg = getattr(self, _Configurable.config_arg)
            if config_arg in kwargs:
                self.config = AttributeDict(kwargs.pop(config_arg))

    @classmethod
    def __subclasshook__(cls, klass: type) -> t.Union[bool, type(NotImplemented)]:
        """Subclass checking hook for types that are alike"""
        if cls is Configurable:
            if all(
                [
                    has_property(klass, "config"),
                    has_property(klass, "configured"),
                    has_property(klass, "required_config_keys"),
                ]
            ):
                return True
        return NotImplemented

    @property
    def required_config_keys(self) -> t.List[str]:
        """Produce a list of required keys"""
        return []

    @property
    def configured(self) -> bool:
        """Is the Configurable configured?"""
        return self.config is not None

    @property
    def config(self) -> t.Optional[AttributeDict]:
        """Access the config"""
        return getattr(self, "_config", None)

    @config.setter
    def config(self, value: t.MutableMapping) -> None:
        """Set the config with key requirements and validation"""
        if not isinstance(value, t.MutableMapping):
            raise TypeError("'config' can be only set with a MutableMapping", value)

        _ = [k for k in self.required_config_keys if k not in value]
        if _:
            raise RequiredKeysMissingError(f"required keys missing: {_!r}")

        self._config = AttributeDict.from_dict(value)
        self.validate()

    def validate(self) -> t.NoReturn:
        """Check if the config has desired values and/or types"""
        pass


"""
HasParent
---------

This mixin allows for easier handling of class hierarchies. A parent type is set and
a :meth:`parent` property is set in an inheriting class.
"""

# shorthands for attribute names
class _HasParent(SimpleNamespace):
    parent_type = read_only("__parent__")
    parent_instance = read_only("_parent")


class HasParent(metaclass=abc.ABCMeta):
    def __init_subclass__(cls, parent: type, **kwargs):
        if not parent and not isinstance(parent, type):
            raise HasParentTypeError("'parent' should be a type", cls)

        setattr(cls, _HasParent.parent_type, parent)

        super().__init_subclass__(**kwargs)

    def __init__(self, *args, **kwargs):
        if "parent" in kwargs:
            self.parent = kwargs.pop("parent")

    @property
    def parent(self):
        return getattr(self, _HasParent.parent_instance, None)

    @parent.setter
    def parent(self, value):
        if not isinstance(value, getattr(self, _HasParent.parent_type)):
            raise WrongParentTypeError(type(value), value)

        setattr(self, _HasParent.parent_instance, value)


"""
SubclassTracker
---------------

Subclass tracker can be used to keep account of derived classes, and concrete
derived classes for any class that inherits from it.

If a type has a member type 'Type' that is an Enum, the tracking will also be
done by type.
"""

# shorthands for attribute names
class _SubclassTracker(SimpleNamespace):
    type_attr = read_only("__type__")
    track = read_only("__track__")
    derived = read_only("__derived__")
    concrete = read_only("__concrete__")
    derived_bt = read_only("__derived_by_type__")
    concrete_bt = read_only("__concrete_by_type__")


class SubclassTracker(metaclass=abc.ABCMeta):
    """A mixin class for tracking derived and concrete derived classes

    .. note::
        The proper functioning of this class depends on the MRO (method
        resolution order). The subclass initialiser looks for itself in the
        hierarchy, and considers the previous class in the instance as the
        "root" to be tracked.

        For example:

        >>> class X(SomeClass, SubclassTracker): pass

        will "apply" the tracking to SomeClass. To track X, put the mixin
        as the first inherited class:

        >>> class X(SubclassTracker, SomeClass): pass
    """

    def __init_subclass__(cls, track: str = None, **kwargs: t.Dict) -> None:
        __ = _SubclassTracker()

        # get root and previous classes
        mixin_idx = cls.mro().index(SubclassTracker)
        root_idx = mixin_idx - 1  # root is the class preceding this mixin
        root = cls.mro()[root_idx]
        prev = cls.mro()[1] if mixin_idx != 1 else None

        if cls is root:
            # set up the root class for tracking
            setgetattr(root, __.derived, default=set())
            setgetattr(root, __.concrete, default=set())
            setgetattr(root, __.track, default=track)

            if has_type(root):
                _rtrack = getattr(root, __.track)
                kwargs_type = kwargs.pop(_rtrack) if _rtrack in kwargs else None
                setgetattr(root, __.type_attr, default=kwargs_type)
                setgetattr(
                    root, __.derived_bt, {k: set() for k in [*getattr(root, "Type"), None]}
                )
                setgetattr(
                    root, __.concrete_bt, {k: set() for k in [*getattr(root, "Type"), None]}
                )

                if not is_abstract(root):
                    getattr(root, __.concrete_bt)[getattr(root, __.type_attr)].add(cls)
        else:
            # bootstrap a derived class
            _root = root.__name__
            _cls = cls.__name__

            getattr(root, __.derived).add(cls)

            if not is_abstract(cls):
                getattr(root, __.concrete).add(cls)

            if has_type(root):
                if not has_type(cls):
                    raise SubclassTrackerTypeError(
                        f"root ({_root}) has Type but {_cls} does not", cls
                    )

                _rtrack = getattr(root, __.track)
                cls_type = getattr(cls, __.type_attr, None)
                prev_type = getattr(prev, __.type_attr, None)
                kwargs_type = kwargs.pop(_rtrack) if _rtrack in kwargs else None

                if cls_type and kwargs_type:
                    raise SubclassTrackerValueError(
                        f"{__.type_attr} and class argument {_rtrack} are both present", cls
                    )

                if not cls_type and prev_type:
                    setattr(cls, __.type_attr, prev_type)
                elif not cls_type and kwargs_type:
                    setattr(cls, __.type_attr, kwargs_type)
                else:  # type of cls already set
                    setgetattr(cls, __.type_attr, None)

                getattr(root, __.derived_bt)[getattr(cls, __.type_attr)].add(cls)

                if not is_abstract(cls):
                    getattr(root, __.concrete_bt)[getattr(cls, __.type_attr)].add(cls)

        super().__init_subclass__(**kwargs)


"""
Serialisable
------------

Base class for serialisation mixins.
"""


class Serialisable(metaclass=abc.ABCMeta):
    """Abstract base class for serialisers"""

    def __init_subclass__(cls, **kwargs):
        """Initialise a subclass

        -   Attributes in `serialise_ignore` will be deleted from the instance before
            serialisation.
        -   Attributes in `serialise_stub` will be 'stubbed', i.e. all leaf nodes will
            be replaced with `None`.
        -   If `serialise_no_parent` is set, a parent object, as defined by HasParent,
            will not be serialised.
        -   If `serialise_inherit` is set, serialise

        A concrete class has to implement the following methods:

        -   def dumps(self) -> t.Union[str, bytes]
        -   def dump(self, file: t.Union[Path, io.TextIOBase]) -> None

        And the following class methods:

        -   def loads(cls, source: t.Union[str, bytes]) -> object
        -   def load(cls, source: t.Union[Path, io.TextIOBase]) -> object

        Args:
            serialise_ignore (t.List[str], optional): list of attributes to ignore
            serialise_stub (t.List[str], optional): list of attributes to stub
            serialise_no_parent (bool, optional): serialise parent class?
            serialise_inherit (bool, optional): inherit serialise settings from base?
        """

        def getter(attr, df):
            if attr in kwargs:
                return kwargs.pop(attr)
            else:
                return df

        cls._serialise_inherit = getter(
            "serialise_inherit", mro_getattr(cls, "_serialise_inherit", True)
        )

        def inherit(attr, df):
            return copy.copy(mro_getattr(cls, attr, df) if cls._serialise_inherit else df)

        cls._serialise_to_ignore: t.Set[str] = inherit("_serialise_to_ignore", set())
        cls._serialise_to_stub: t.Set[str] = inherit("_serialise_to_stub", set())
        cls._serialise_to_save: t.Set[str] = inherit("_serialise_to_save", set())
        cls._serialise_io_type: str = inherit(
            "_serialise_io_type", getter("serialise_io_type", "text")
        )
        cls._serialise_no_parent = inherit(
            "_serialise_no_parent", getter("serialise_no_parent", True)
        )
        cls._serialise_sep = "."

        serialise_ignore = getter("serialise_ignore", [])
        serialise_stub = getter("serialise_stub", [])
        serialise_save = getter("serialise_save", [])

        if not isinstance(cls._serialise_io_type, str) and not any(
            (cls._serialise_io_type == "text", cls._serialise_io_type == "binary")
        ):
            raise TypeError(
                f"{serialise_io_type!r} is not a recognised io type (either 'text' "
                "or 'binary')"
            )

        if cls._serialise_no_parent:
            cls._serialise_to_ignore.add(_HasParent.parent_instance)

        for attr_to_ignore in serialise_ignore:
            if not isinstance(attr_to_ignore, str):
                raise SerialisableValueError(
                    "'serialise_ignore' should only have string objects"
                )

            cls._serialise_to_ignore.add(attr_to_ignore)

        for attr_to_stub in serialise_stub:
            if not isinstance(attr_to_stub, str):
                raise SerialisableValueError("'serialise_stub' should only have string objects")

            cls._serialise_to_stub.add(attr_to_stub)

        for attr_to_save in serialise_save:
            if not isinstance(attr_to_save, str):
                raise SerialisableValueError("'serialise_save' should only have string objects")
            if cls._serialise_sep in attr_to_save:
                raise SerialisableValueError(
                    "the attribute to save contains the reserved special character "
                    "{!r} used in the serialisation process".format(cls._serialise_sep)
                )

            cls._serialise_to_save.add(attr_to_save)

        # Check all pairwise combinations for conflicts
        intersections = map(
            lambda combination: set.intersection(*map(set, combination)),
            itertools.combinations([serialise_ignore, serialise_stub, serialise_save], 2),
        )
        if not all([not len(_) for _ in intersections]):
            raise SerialisableValueError(
                "'serialise_ignore', 'serialise_stub', and 'serialise_save' should be disjoint",
                serialise_ignore,
                serialise_stub,
                serialise_save,
            )

        super().__init_subclass__(**kwargs)

    @abc.abstractmethod
    def dumps(self) -> t.Union[str, bytes]:
        """Dump an instance to a string or bytes"""
        pass

    @abc.abstractmethod
    def dump(self, file: io.IOBase) -> None:
        """Dump an instance to a file

        Args:
            file (t.Union[Path, io.TextIOBase]): a file path or an io wrapper
        """
        pass

    @classmethod
    @abc.abstractmethod
    def loads(cls, source: t.Union[str, bytes]) -> object:
        """Load an instance from a string or bytes

        Args:
            source (t.Union[str, bytes]): a serialised object as a string or bytes
        """
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, source: io.IOBase) -> object:
        """Load an instance from a file

        Args:
            source (t.Union[Path, io.TextIOBase]): a file path or an io wrapper
        """
        pass

    @classmethod
    def deserialise(cls, source=t.Union[str, bytes, Path, t.IO]) -> object:
        local: bool = False
        valid_types = [str, bytes, Path, t.IO]
        if not any(isinstance(source, v) for v in valid_types):
            raise SerialisableTypeError(f"'source' must be one of {valid_types!r}", source)

        if isinstance(source, Path):
            if cls._serialise_io_type == "text":
                source = source.open("r")
            else:
                source = source.open("rb")
            local = True

        if isinstance(source, io.IOBase):
            assert source.readable()
            if cls._serialise_io_type == "text":
                assert isinstance(source, io.TextIOBase)
            else:
                assert isinstance(source, io.BufferedIOBase)

        if isinstance(source, (str, bytes)):
            instance = cls.loads(source)
        else:
            instance = cls.load(source)

        if local:
            source.close()

        assert isinstance(instance, cls)
        return instance

    def serialise(self, file: t.Optional[t.Union[Path, t.IO]] = None) -> t.Optional[str]:
        local: bool = False

        if file:
            if isinstance(file, Path):
                assert check_access(file, "w"), "'file' must be writable"
                if self._serialise_io_type == "text":
                    file = file.open("w")
                else:
                    file = file.open("wb")
                local = True
            elif isinstance(file, io.IOBase):
                assert file.writable()
                if self._serialise_io_type == "text":
                    assert isinstance(file, io.TextIOBase)
                else:
                    assert isinstance(file, io.BufferedIOBase)
            else:
                raise SerialisableTypeError("'file' not one of [str, Path, t.IO]", file)

        serialise_backup = dict()

        for k in self._serialise_to_ignore:
            if hasattr(self, k):
                serialise_backup[k] = getattr(self, k)
                delattr(self, k)

        for k in self._serialise_to_stub:
            if hasattr(self, k):
                serialise_backup[k] = getattr(self, k)
                setattr(self, k, stub_recursively(getattr(self, k)))

        for k in self._serialise_to_save:
            if hasattr(self, k):
                serialise_backup[k] = getattr(self, k)

                # if a mapping is given, stub it, otherwise delete the attribute before serialising
                if isinstance(getattr(self, k), t.Mapping):
                    setattr(self, k, stub_recursively(getattr(self, k)))
                else:
                    delattr(self, k)

        try:
            if file:
                self.dump(file)
            else:
                return self.dumps()
        finally:
            if local:
                file.close()

            for k, v in serialise_backup.items():
                setattr(self, k, v)

    def _save_npz(self, npz_path, npz_data):
        np.savez_compressed(npz_path, **npz_data)
        return npz_path

    def _save_hdf(self, hdf_path, hdf_data):
        with pd.HDFStore(hdf_path, "w") as _pd_store:
            for name, values in hdf_data.items():
                _pd_store.put(name, values)
        return hdf_path

    def save_data(
        self, *, prefix: t.Optional[str] = "", path: t.Optional[Path] = None
    ) -> t.List[Path]:
        """Save input and output data in NumPy archives and/or HDF5 stores.

        Args:
            prefix (t.Optional[str], optional):
                The file prefix, e.g. "my_data" will produce archives my_data.npz and my_data.h5
            path (t.Optional[Path], optional):
                The save path, if different than self.path
        """
        if not path and not hasattr(self, "path"):
            raise ValueError(
                "Path has to be either available as an attribute or provided to the "
                "'load_data' function."
            )

        _saved_paths = []

        _filepath = path if path and path.is_dir() else self.path
        for attribute in self._serialise_to_save:
            if getattr(self, attribute, None) is None:
                continue

            if isinstance(getattr(self, attribute), np.ndarray):
                _saved_paths.append(
                    self._save_npz(
                        _filepath / (prefix + attribute + ".npz"),
                        {attribute: getattr(self, attribute)},
                    )
                )
            elif isinstance(getattr(self, attribute), (pd.DataFrame, pd.Series, pd.Panel)):
                _saved_paths.append(
                    self._save_hdf(
                        _filepath / (prefix + attribute + ".h5"),
                        {attribute: getattr(self, attribute)},
                    )
                )
            elif isinstance(getattr(self, attribute), t.Mapping):
                _saved_paths.append(
                    self.save_mapping(attribute, prefix, path, save_bundled=True)
                )
            else:
                raise SerialisableDataSaveError(
                    f"Failed saving {attribute} from 'serialise_to_save', "
                    "can only save NumPy or Pandas data!"
                )

        return _saved_paths

    def save_mapping(
        self,
        attribute,
        prefix: t.Optional[str] = "",
        path: t.Optional[Path] = None,
        save_bundled: t.Optional[bool] = False,
    ) -> t.List[Path]:
        _saved_paths = list()
        _flattened_holder = flatten_dict(getattr(self, attribute), self._serialise_sep)
        _filepath = path if path and path.is_dir() else self.path

        if not _filepath.exists():
            _filepath.mkdir()
        if save_bundled:
            _np_holder = dict()
            _pd_holder = dict()
            for name, value in _flattened_holder.items():
                if isinstance(value, np.ndarray):
                    if name in _np_holder:
                        raise SerialisableDataSaveError(
                            f"{name!r} already contained in NumPy attribute holder {_np_holder!r}"
                        )
                    _np_holder[name] = value
                elif isinstance(value, (pd.DataFrame, pd.Series, pd.Panel)):
                    if name in _pd_holder:
                        raise SerialisableDataSaveError(
                            f"{name!r} already contained in Pandas attribute holder {_pd_holder!r}"
                        )
                    _pd_holder[name] = value
                else:
                    raise SerialisableDataSaveError(
                        f"type {type(value)!r} in {name!r} cannot be serialised"
                    )

            if _np_holder:
                _np_path = _filepath / (prefix + attribute + ".npz")
                np.savez_compressed(_np_path, **_np_holder)
                _saved_paths.append(_np_path)

            if _pd_holder:
                _pd_path = _filepath / (prefix + attribute + ".h5")
                with pd.HDFStore(_pd_path, "w") as _pd_store:
                    for name, value in _pd_holder.items():
                        _pd_store.put(name, value)
                _saved_paths.append(_pd_path)
        else:
            for name, value in _flattened_holder.items():
                if isinstance(value, np.ndarray):
                    _saved_paths.append(
                        self._save_npz(_filepath / (prefix + name + ".npz"), {name: value})
                    )
                elif isinstance(value, (pd.DataFrame, pd.Series, pd.Panel)):
                    _saved_paths.append(
                        self._save_hdf(_filepath / (prefix + name + ".h5"), {name: value})
                    )
                else:
                    # raise SerialisableDataSaveError(
                    get_root_logger().info(
                        f"type {type(value)!r} in {name!r} cannot be serialised"
                    )

        return _saved_paths

    def remove_mapping(
        self,
        attribute,
        prefix: t.Optional[str] = "",
        path: t.Optional[Path] = None,
        saved_bundled: t.Optional[bool] = False,
    ) -> t.List[Path]:
        _removed_paths = list()
        _flattened_holder = flatten_dict(getattr(self, attribute), self._serialise_sep)
        _filepath = path if path and path.is_dir() else self.path

        if not _filepath.exists():
            _filepath.mkdir()
        if saved_bundled:
            _np_path = _filepath / (prefix + attribute + ".npz")
            if _np_path.exists():
                os.remove(_np_path)
                _saved_paths.append(_np_path)

            _pd_path = _filepath / (prefix + attribute + ".h5")
            if _pd_path.exists():
                os.remove(_pd_path)
                _removed_paths.append(_pd_path)
        else:
            for name, value in _flattened_holder.items():
                _np_path = _filepath / (prefix + name + ".npz")
                if _np_path.exists():
                    os.remove(_np_path)
                    _removed_paths.append(_np_path)

                _pd_path = _filepath / (prefix + name + ".h5")
                if _pd_path.exists():
                    os.remove(_pd_path)
                    _removed_paths.append(_pd_path)
        return _removed_paths

    def save_data_bundled(
        self, *, prefix: t.Optional[str] = None, path: t.Optional[Path] = None
    ) -> t.List[Path]:
        """Save input and output data in NumPy archives and/or HDF5 stores.

        Args:
            prefix (t.Optional[str], optional):
                The file prefix, e.g. "my_data" will produce archives my_data.npz and my_data.h5
            path (t.Optional[Path], optional):
                The save path, if different than self.path
        """
        if not path and not hasattr(self, "path"):
            raise ValueError(
                "Path has to be either available as an attribute or provided to the "
                "'load_data' function."
            )

        _np_holder = dict()
        _pd_holder = dict()
        _mapping_holder = dict()

        _saved_paths = []
        _np_file = prefix + ".npz" if prefix else "streams.npz"
        _np_path = path / _np_file if path and path.is_dir() else self.path / _np_file
        _pd_file = prefix + ".h5" if prefix else "streams.h5"
        _pd_path = path / _pd_file if path and path.is_dir() else self.path / _pd_file

        for attribute in self._serialise_to_save:
            if getattr(self, attribute, None) is None:
                continue

            if isinstance(getattr(self, attribute), np.ndarray):
                _np_holder[attribute] = getattr(self, attribute)
            elif isinstance(getattr(self, attribute), (pd.DataFrame, pd.Series, pd.Panel)):
                _pd_holder[attribute] = getattr(self, attribute)
            elif isinstance(getattr(self, attribute), t.Mapping):
                _mapping_holder[attribute] = getattr(self, attribute)
            else:
                raise SerialisableDataSaveError(
                    f"Failed saving {attribute} from 'serialise_to_save', "
                    "can only save NumPy or Pandas data!"
                )

        # For all serialisable-save attributes that were mappings, flatten them and assign to
        # NumPy and Pandas holders.
        if _mapping_holder:
            _flattened_holder = flatten_dict(_mapping_holder, self._serialise_sep)
            for name, value in _flattened_holder.items():
                if isinstance(value, np.ndarray):
                    if name in _np_holder:
                        raise SerialisableDataSaveError(
                            f"{name!r} already contained in NumPy attribute holder {_np_holder!r}"
                        )
                    _np_holder[name] = value
                elif isinstance(value, (pd.DataFrame, pd.Series, pd.Panel)):
                    if name in _pd_holder:
                        raise SerialisableDataSaveError(
                            f"{name!r} already contained in Pandas attribute holder {_pd_holder!r}"
                        )
                    _pd_holder[name] = value
                else:
                    raise SerialisableDataSaveError(
                        f"type {type(value)!r} in {name!r} cannot be serialised"
                    )

        if _np_holder:
            np.savez_compressed(_np_path, **_np_holder)
            _saved_paths.append(_np_path)

        if _pd_holder:
            with pd.HDFStore(_pd_path, "w") as _pd_store:
                for name, value in _pd_holder.items():
                    _pd_store.put(name, value)
            _saved_paths.append(_pd_path)

        return _saved_paths

    def load_data(self, *, prefix: t.Optional[str] = "", path: t.Optional[Path] = None) -> None:
        """Loads input and output stream data from NumPy archives and/or HDF5 stores.

        Args:
            prefix (t.Optional[str], optional):
                The file prefix, e.g. "my_data" will produce archives my_data.npz and my_data.h5
            path (t.Optional[Path], optional):
                The load path, if different than self.path
        """
        if not path and not hasattr(self, "path"):
            raise ValueError(
                "Path has to be either available as an attribute or provided to the "
                "'load_data' function."
            )

        # Get names and valid paths of
        stubbed = {
            _: list(
                get_valid_access_paths(
                    getattr(self, _),
                    _leaf_only=True,
                    _use_lists=False,
                    _fallthrough_empty=False,
                )
            )
            for _ in self._serialise_to_save
            if hasattr(self, _) and isinstance(getattr(self, _), t.Mapping)
        }

        _filepath = path if path and path.is_dir() else self.path
        for attribute in self._serialise_to_save:
            _np_path = _filepath / (prefix + attribute + ".npz")
            _pd_path = _filepath / (prefix + attribute + ".h5")
            if _np_path.exists():
                with np.load(_np_path) as _np_store:
                    for _data in _np_store.files:
                        sub_attribute, delimiter, rest = _data.partition(self._serialise_sep)
                        setattr(self, _data, _np_store[_data])

            elif _pd_path.exists():
                with pd.HDFStore(_pd_path, "r") as _pd_store:
                    for _data in _pd_store:
                        sub_attribute, delimiter, rest = _data.strip("/").partition(
                            self._serialise_sep
                        )
                        setattr(self, attribute, _pd_store.get(_data))
            else:
                get_root_logger().warning(f"No file found to load attribute{attribute!r}!")

    def load_mapping(self, path: Path, prefix: t.Optional[str] = None) -> AttributeDict:
        """
        Loads data from files specified using path and regex. Different to load_data, the data is not saved
        in the parent object (self) but in a AttributeDict, which is then returned.
        """
        if not path:
            raise ValueError("Path has to be provided to the 'load_mapping' function.")

        def _load_data_file(_data_file_path, _fileextension):
            if _fileextension == "npz":
                with np.load(_data_file_path) as _np_store:
                    if len(_np_store) == 1:
                        _data = _np_store[_np_store.files[0]]
                    else:
                        _data = AttributeDict()
                        for data_key in _np_store:
                            _data[data_key] = _np_store[data_key]
            elif _fileextension == "h5":
                with pd.HDFStore(_data_file_path, "r") as _pd_store:
                    if len(_pd_store) == 1:
                        _data = _pd_store.get(_pd_store.keys()[0])
                    else:
                        _data = AttributeDict()
                        for data_key in _pd_store:
                            _data[data_key] = _pd_store[data_key]
            else:
                raise Exception(f"Unknown file extension for file {_data_file_path}")
            return _data

        data = AttributeDict()
        for filepath in path.glob(prefix + "*"):
            filename = filepath.name.replace(prefix, "")
            filename_split = filename.split(".")
            if len(filename_split) == 2:
                attribute = filename_split[0]
                data[attribute] = _load_data_file(filepath, filename_split[-1])
            elif len(filename_split) == 3:
                attribute_outer = filename_split[0]
                attribute_inner = filename_split[1]
                if attribute_outer not in data.keys():
                    data[attribute_outer] = AttributeDict()
                data[attribute_outer][attribute_inner] = _load_data_file(
                    filepath, filename_split[-1]
                )
            else:
                raise Exception(
                    f"Filename {filename} is not fitting. Has to consist of two or three comma separated substrings"
                )
        return data

    def load_data_bundled(
        self, *, prefix: t.Optional[str] = None, path: t.Optional[Path] = None
    ) -> None:
        """Loads input and output stream data from NumPy archives and/or HDF5 stores.

        Args:
            prefix (t.Optional[str], optional):
                The file prefix, e.g. "my_data" will produce archives my_data.npz and my_data.h5
            path (t.Optional[Path], optional):
                The load path, if different than self.path
        """
        if not path and not hasattr(self, "path"):
            raise ValueError(
                "Path has to be either available as an attribute or provided to the "
                "'load_data' function."
            )

        _np_file = prefix + ".npz" if prefix else "streams.npz"
        _pd_file = prefix + ".h5" if prefix else "streams.h5"
        _np_path = path / _np_file if path and path.is_dir() else self.path / _np_file
        _pd_path = path / _pd_file if path and path.is_dir() else self.path / _pd_file

        # Get names and valid paths of
        stubbed = {
            _: list(
                get_valid_access_paths(
                    getattr(self, _),
                    _leaf_only=True,
                    _use_lists=False,
                    _fallthrough_empty=False,
                )
            )
            for _ in self._serialise_to_save
            if hasattr(self, _) and isinstance(getattr(self, _), t.Mapping)
        }

        if _np_path.exists():
            with np.load(_np_path) as _np_store:
                for _data in _np_store.files:
                    attribute, delimiter, rest = _data.partition(self._serialise_sep)
                    if delimiter and attribute in stubbed:
                        # Split the 'rest' and form an access query tuple, check if it's valid
                        query = tuple(rest.split(self._serialise_sep))
                        if query not in stubbed[attribute]:
                            raise SerialisableDataLoadError(
                                f"Access query {query!r} not valid for attribute{attribute!r}"
                            )

                        setitem(getattr(self, attribute), query, _np_store[_data])
                    else:
                        setattr(self, _data, _np_store[_data])

        if _pd_path.exists():
            with pd.HDFStore(_pd_path, "r") as _pd_store:
                for _data in _pd_store:
                    attribute, delimiter, rest = _data.strip("/").partition(self._serialise_sep)
                    if delimiter and attribute in stubbed:
                        # Split the 'rest' and form an access query tuple, check if it's valid
                        query = tuple(rest.split(self._serialise_sep))
                        if query not in stubbed[attribute]:
                            raise SerialisableDataLoadError(
                                f"Access query {query!r} not valid for attribute{attribute!r}"
                            )

                        setitem(getattr(self, attribute), query, _pd_store.get(_data))
                    elif attribute in self._serialise_to_save:
                        setattr(self, attribute, _pd_store.get(_data))


"""
Pickleable
----------

Implementation of Serialisable for the pickle binary serialisation format.
"""


class Pickleable(Serialisable, serialise_io_type="binary"):
    pickle_version = read_only(-1)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def dumps(self) -> t.Union[str, bytes]:
        return pickle.dumps(self, self.pickle_version)

    def dump(self, file: io.IOBase) -> None:
        assert isinstance(file, io.BufferedIOBase), "expects a buffered I/O"
        pickle.dump(self, file, self.pickle_version)

    @classmethod
    def loads(cls, source: t.Union[str, bytes]) -> object:
        return pickle.loads(source)

    @classmethod
    def load(cls, source: io.IOBase) -> object:
        assert isinstance(source, io.BufferedIOBase), "expects a buffered I/O"
        return pickle.load(source)


"""
YamlSerialisable
----------------

Implementation of Serialisable for YAML using the PyYAML library.
"""


class YamlSerialisable(Serialisable, serialise_io_type="text"):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def dumps(self) -> t.Union[str, bytes]:
        return yaml.dump(self, Dumper=Dumper)

    def dump(self, file: t.Union[Path, io.TextIOBase]) -> None:
        yaml.dump(self, file, Dumper=Dumper)

    @classmethod
    def loads(cls, source: t.Union[str, bytes]) -> object:
        return yaml.load(source, Loader=Loader)

    @classmethod
    def load(cls, source: t.Union[Path, io.TextIOBase]) -> object:
        return yaml.load(source, Loader=Loader)


"""
Intermediates
-------------

Factory for class member variables that are used to save intermediate results needed for closer
data inspection.
"""


class IntermediatesHolder(metaclass=abc.ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if mro_hasattr(cls, "_serialise_to_ignore"):
            cls._serialise_to_ignore.add("intermediates")

    def __init__(self, *args, **kwargs):
        self._intermediates = AttributeDict()

    @property
    def intermediates(self):
        if not hasattr(self, "_intermediates"):
            self._intermediates = AttributeDict()
        return getattr(self, "_intermediates")

    @intermediates.setter
    def intermediates(self, value):
        if not isinstance(value, t.Mapping):
            raise IntermediatesTypeError("intermediates can only be set to mappings")

        setattr(
            self,
            "_intermediates",
            value if isinstance(value, AttributeDict) else AttributeDict.from_dict(value),
        )

    @intermediates.deleter
    def intermediates(self):
        delattr(self, "_intermediates")

    def add_intermediate(self, name, value):
        self.intermediates[name] = value

    def del_intermediate(self, name):
        if name in self.intermediates:
            del self.intermediates[name]

    def clear_intermediates(self):
        self.intermediates.clear()

    def get_intermediate(self, name):
        return self.intermediates.get(name, None)


class IntermediatesHandler(IntermediatesHolder):
    @staticmethod
    def _is_intermediate_holder(value):
        if hasattr(value, "intermediates"):
            return True
        else:
            return False

    def collect_intermediates(self, source: object, namespace: t.Optional[str] = None):
        if isinstance(source, t.Mapping):
            candidates = {
                k: v
                for k, v in source.items()
                if self._is_intermediate_holder(v) and len(v.intermediates)
            }

            copies = {k: v.intermediates.copy() for k, v in candidates.items()}

            if namespace:
                self.intermediates[namespace] = AttributeDict.from_dict(copies)
            else:
                self.intermediates.update(copies)

            # Collected layers need to be cleared to prevent unnecessary serialisation:
            # IntermediatesHolder instances usually do not derive from any Serialisable
            # subclasses.
            for name, obj in candidates.items():
                obj.clear_intermediates()
        else:
            raise TypeError(
                "Can only collect intermediates from mappings, got: {!r}".format(type(source))
            )


"""
Conversion
----------

This class can be used to convert unit labels according to powers of 10 prefixes
"""


class LabelConversions(metaclass=abc.ABCMeta):
    @property
    def _unit_prefixes(self):
        return {
            0.000000000000000000000000000000000000000000000000000001: {
                "name": "Septendecillionth",
                "power": -54,
                "symbol": "",
                "prefix": "",
            },
            0.000000000000000000000000000000000000000000000000001: {
                "name": "Sedecillionth",
                "power": -51,
                "symbol": "",
                "prefix": "",
            },
            0.000000000000000000000000000000000000000000000001: {
                "name": "Quindecillionth",
                "power": -48,
                "symbol": "",
                "prefix": "",
            },
            0.000000000000000000000000000000000000000000001: {
                "name": "Quattuordecillionth",
                "power": -45,
                "symbol": "",
                "prefix": "",
            },
            0.000000000000000000000000000000000000000001: {
                "name": "Tredecillionth",
                "power": -42,
                "symbol": "",
                "prefix": "",
            },
            0.000000000000000000000000000000000000001: {
                "name": "Duodecillionth",
                "power": -39,
                "symbol": "",
                "prefix": "",
            },
            0.000000000000000000000000000000000001: {
                "name": "Undecillionth",
                "power": -36,
                "symbol": "",
                "prefix": "",
            },
            0.000000000000000000000000000000001: {
                "name": "Decillionth",
                "power": -33,
                "symbol": "",
                "prefix": "",
            },
            0.000000000000000000000000000001: {
                "name": "Nonillionth",
                "power": -30,
                "symbol": "",
                "prefix": "",
            },
            0.000000000000000000000000001: {
                "name": "Octillionth",
                "power": -27,
                "symbol": "",
                "prefix": "",
            },
            0.000000000000000000000001: {
                "name": "Septillionth",
                "power": -24,
                "symbol": "y",
                "prefix": "Yocto",
            },
            0.000000000000000000001: {
                "name": "Sextillionth",
                "power": -21,
                "symbol": "z",
                "prefix": "Zepto",
            },
            0.000000000000000001: {
                "name": "Quintillionth",
                "power": -18,
                "symbol": "a",
                "prefix": "Atto",
            },
            0.000000000000001: {
                "name": "Quadrillionth",
                "power": -15,
                "symbol": "f",
                "prefix": "Femto",
            },
            0.000000000001: {
                "name": "Trillionth",
                "power": -12,
                "symbol": "p",
                "prefix": "Pico",
            },
            0.000000001: {"name": "Billionth", "power": -9, "symbol": "n", "prefix": "Nano"},
            0.000001: {"name": "Millionth", "power": -6, "symbol": "μ", "prefix": "Micro"},
            0.00001: {
                "name": "Hundred-thousandth (Lakh|Lacth)",
                "power": -5,
                "symbol": "",
                "prefix": "",
            },
            0.0001: {
                "name": "Ten-thousandth (Myriadth)",
                "power": -4,
                "symbol": "",
                "prefix": "",
            },
            0.001: {"name": "Thousandth", "power": -3, "symbol": "m", "prefix": "Milli"},
            0.01: {"name": "Hundredth", "power": -2, "symbol": "c", "prefix": "Centi"},
            0.1: {"name": "Tenth", "power": -1, "symbol": "d", "prefix": "Deci"},
            1: {"name": "One", "power": 0, "symbol": "", "prefix": ""},
            10: {"name": "Ten", "power": 1, "symbol": "D", "prefix": "Deca"},
            100: {"name": "Hundred", "power": 2, "symbol": "H", "prefix": "Hecto"},
            1000: {"name": "Thousand", "power": 3, "symbol": "K", "prefix": "Kilo"},
            10000: {"name": "Ten Thousand (Myriad)", "power": 4, "symbol": "", "prefix": ""},
            100000: {"name": "Hundred Thousand (Lakh)", "power": 5, "symbol": "", "prefix": ""},
            1000000: {"name": "Million", "power": 6, "symbol": "M", "prefix": "Mega"},
            10000000: {"name": "Ten Million (Crore)", "power": 7, "symbol": "", "prefix": ""},
            100000000: {"name": "Hundred Million", "power": 8, "symbol": "", "prefix": ""},
            1000000000: {
                "name": "Billion (Milliard)",
                "power": 9,
                "symbol": "G",
                "prefix": "Giga",
            },
            1000000000000: {
                "name": "Trillion (Billion)",
                "power": 12,
                "symbol": "T",
                "prefix": "Tera",
            },
            1000000000000000: {
                "name": "Quadrillion (Billiard)",
                "power": 15,
                "symbol": "P",
                "prefix": "Peta",
            },
            1000000000000000000: {
                "name": "Quintillion (Trillion)",
                "power": 18,
                "symbol": "E",
                "prefix": "Exa",
            },
            1000000000000000000000: {
                "name": "Sextillion (Trilliard)",
                "power": 21,
                "symbol": "Z",
                "prefix": "Zetta",
            },
            1000000000000000000000000: {
                "name": "Septillion (Quadrillion)",
                "power": 24,
                "symbol": "Y",
                "prefix": "Yotta",
            },
            1000000000000000000000000000: {
                "name": "Octillion (Quadrilliard)",
                "power": 27,
                "symbol": "",
                "prefix": "",
            },
            1000000000000000000000000000000: {
                "name": "Nonillion (Quintillion)",
                "power": 30,
                "symbol": "",
                "prefix": "",
            },
            1000000000000000000000000000000000: {
                "name": "Decillion (Quintilliard)",
                "power": 33,
                "symbol": "",
                "prefix": "",
            },
            1000000000000000000000000000000000000: {
                "name": "Undecillion (Sextillion)",
                "power": 36,
                "symbol": "",
                "prefix": "",
            },
            1000000000000000000000000000000000000000: {
                "name": "Duodecillion (Sextilliard)",
                "power": 39,
                "symbol": "",
                "prefix": "",
            },
            1000000000000000000000000000000000000000000: {
                "name": "Tredecillion (Septillion)",
                "power": 42,
                "symbol": "",
                "prefix": "",
            },
            1000000000000000000000000000000000000000000000: {
                "name": "Quattuordecillion (Septilliard)",
                "power": 45,
                "symbol": "",
                "prefix": "",
            },
            1000000000000000000000000000000000000000000000000: {
                "name": "Quindecillion (Octillion)",
                "power": 48,
                "symbol": "",
                "prefix": "",
            },
            1000000000000000000000000000000000000000000000000000: {
                "name": "Sexdecillion (Octilliard)",
                "power": 51,
                "symbol": "",
                "prefix": "",
            },
            1000000000000000000000000000000000000000000000000000000: {
                "name": "Septendecillion (Nonillion)",
                "power": 54,
                "symbol": "",
                "prefix": "",
            },
            1000000000000000000000000000000000000000000000000000000000000000: {
                "name": "Vigintillion (Decilliard)",
                "power": 63,
                "symbol": "",
                "prefix": "",
            },
            10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000: {
                "name": "Googol",
                "power": 100,
                "symbol": "",
                "prefix": "",
            },
        }

    def _label_unit_conversion(self, label, factor):
        label_unit_symbol = label.split("::")[-1][0]
        for lbl_factor in self._unit_prefixes:
            if self._unit_prefixes[lbl_factor]["symbol"] == label_unit_symbol:
                new_label_unit_symbol = self._unit_prefixes[lbl_factor * factor]["symbol"]
                break
        if len(new_label_unit_symbol) == 0 and (lbl_factor * factor) != 1:
            new_label_unit_symbol = self._unit_prefixes[lbl_factor * factor]["name"] + " "
        return label.replace("::" + label_unit_symbol, "::" + new_label_unit_symbol)
