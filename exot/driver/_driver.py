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
"""Driver base class"""

from __future__ import annotations

import typing as t
from contextlib import AbstractContextManager

from exot.exceptions import *
from exot.util.logging import get_root_logger
from exot.util.misc import is_abstract
from exot.util.mixins import SubclassTracker

from ._backend import Backend
from ._mixins import *

__all__ = ("Driver",)


class Driver(
    SubclassTracker,
    TransferMethodsMixin,
    ProcessMethodsMixin,
    FileMethodsMixin,
    LockingMethodsMixin,
    AbstractContextManager,
):
    _backend_type: t.Type[Backend]

    DEFAULT_FETCH_EXCLUDE_LIST = ["*.pickle", "*.npz", "*.h5", "*.toml"]

    DEFAULT_SEND_EXCLUDE_LIST = [
        "*.pickle",
        "*.npz",
        "*.h5",
        "*.log.csv",
        "*.debug.txt",
        "_logs",
    ]

    def __init_subclass__(cls, backend: Backend, **kwargs: t.Any) -> None:
        if not isinstance(backend, type):
            raise DriverTypeError("driver backend not a type", type(backend), type)

        if not issubclass(backend, Backend):
            raise DriverTypeError("wrong driver backend", type(backend), Backend)

        if is_abstract(backend):
            raise DriverTypeError(f"backend cannot be abstract")

        cls._backend_type = backend

        super().__init_subclass__(**kwargs)

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        self._backend = self._backend_type(*args, **kwargs)

        setable_states_props = [hasattr(self.__class__, state) for state in self.setable_states]
        getable_states_props = [hasattr(self.__class__, state) for state in self.getable_states]

        if self.setable_states and not all(setable_states_props):
            raise TypeError(
                "some of the setable states are not available as attributes",
                dict(zip(self.setable_states, setable_states_props)),
            )

        if self.getable_states and not all(getable_states_props):
            raise TypeError(
                "some of the getable states are not available as attributes",
                dict(zip(self.getable_states, getable_states_props)),
            )

        for prop in self.setable_states:
            if hasattr(self.__class__, prop):
                if isinstance(getattr(self.__class__, prop), property):
                    if getattr(self.__class__, prop).fset is None:
                        raise TypeError("setable property does not have a setter", prop)

        for prop in self.setable_states:
            if hasattr(self.__class__, prop):
                if isinstance(getattr(self.__class__, prop), property):
                    if getattr(self.__class__, prop).fget is None:
                        raise TypeError("getable property does not have a getter", prop)

    def __enter__(self) -> Driver:
        if not self.backend.connected:
            self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        return "<{}.{} at {} ({}, {})>".format(
            self.__module__,
            self.__class__.__name__,
            hex(id(self)),
            "configured" if self.backend.configured else "not configured",
            "active" if self.connected else "inactive",
        )

    @classmethod
    def instantiate(cls, *args, **kwargs) -> Driver:
        """Get a connected Driver instance"""
        instance = cls(*args, **kwargs)
        if not instance.can_connect():
            raise CouldNotConnectError("cannot connect")
        instance.connect()
        return instance

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def connection_id(self) -> str:
        if self.backend.configured:
            return self.backend.config.ip

    @property
    def connected(self) -> bool:
        return self.backend.connected

    def can_connect(self) -> bool:
        return self.backend.can_connect()

    def connect(self, force=False) -> None:
        assert self.backend.can_connect(), "driver's backend not configured"
        self.backend.connect()

        if (not force) and self.is_locked:
            raise PlatformLocked(f"Zone with ip: {self.backend.config.ip} was locked")

        self.lock()

        if not self.original_state:
            self._original_state = self.getstate()
            self.initstate()

    def disconnect(self) -> None:
        if self.backend.connected:
            self.unlock()

            self.backend.disconnect()

    def command(
        self, cmd: Backend.CommandT, sudo: bool = False, **kwargs: dict
    ) -> Backend.ReturnT:
        assert self.backend, "driver backend must exist"
        return self.backend.sudo(cmd, **kwargs) if sudo else self.backend.run(cmd, **kwargs)

    @property
    def setable_states(self) -> t.List[str]:
        return []

    @property
    def getable_states(self) -> t.List[str]:
        return []

    def getstate(self) -> t.Dict:
        return {state: getattr(self, state) for state in self.getable_states}

    def setstate(self, **kwargs) -> None:
        for state in kwargs:
            if state in self.setable_states:
                setattr(self, state, kwargs.get(state, None))

    def initstate(self) -> None:
        pass

    @property
    def original_state(self) -> t.Dict:
        return getattr(self, "_original_state", None)

    def cleanup(self) -> None:
        _ = self.original_state.copy()
        self.setstate(**_)
