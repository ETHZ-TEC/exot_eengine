"""Driver base class"""

from __future__ import annotations

import typing as t
from contextlib import AbstractContextManager

from exot.exceptions import *
from exot.util.misc import is_abstract
from exot.util.mixins import SubclassTracker

from ._backend import Backend
from ._mixins import *

__all__ = ("Driver",)


"""
Driver
------

Synopsis & signatures::

access       (self, path: str, mode: str = 'r') -> bool
can_connect  (self) -> 'bool'
cleanup      (self)
command      (self, cmd: 'Backend.CommandT', **kwargs: 'dict') -> 'Backend.ReturnT'
connect      (self) -> 'None'
copy         (self, path_from: str, path_to: str, recursive: bool = True, overwrite: bool = True) -> bool
delete       (self, path: str, recursive: bool = False) -> str
disconnect   (self) -> 'None'
exists       (self, path: str) -> bool
fetch        (self, path_from: pathlib.Path, path_to: pathlib.Path) -> bool
getstate     (self) -> Dict
instantiate  (*args, **kwargs) -> 'Driver'
is_dir       (self, path: str) -> bool
is_file      (self, path: str) -> bool
kill         (self, pid: int) -> bool
mkdir        (self, path: str, parents: bool = True) -> bool
move         (self, path_from: str, path_to: str, overwrite: bool = True) -> bool
send         (self, path_from: pathlib.Path, path_to: pathlib.Path) -> bool
setstate     (self, value: Dict)
signal       (self, pid: int, sig: Union[str, int], *args) -> bool
spawn        (self, path: str, args: List[str]) -> Tuple[int, invoke.runners.Result]
start        (self, pid: int) -> bool
stop         (self, pid: int) -> bool
"""


class Driver(
    SubclassTracker,
    TransferMethodsMixin,
    ProcessMethodsMixin,
    FileMethodsMixin,
    SetupMethodsMixin,
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
    def original_state(self) -> t.Dict:
        return getattr(self, "_original_state", None)

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def connected(self) -> bool:
        return self.backend.connected

    def can_connect(self) -> bool:
        return self.backend.can_connect()

    def connect(self, force=False) -> None:
        assert self.backend.can_connect(), "driver's backend not configured"
        self.backend.connect()

        if (not force) and self.is_locked:
            raise PlatformLocked(f"Platform with ip: {self.backend.config.ip} was locked")

        self.lock()

        if not self.original_state:
            self._original_state = self.getstate()

    def disconnect(self) -> None:
        if self.backend.connected:
            self.unlock()

            self.backend.disconnect()

    def command(self, cmd: Backend.CommandT, **kwargs: dict) -> Backend.ReturnT:
        assert self.backend, "driver backend must exist"
        return self.backend.run(cmd, **kwargs)
