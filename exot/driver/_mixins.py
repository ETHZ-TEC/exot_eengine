"""Abstract mixin classes to separate concerns and define interfaces"""

import abc
import os
import typing as t
from pathlib import Path

from ._backend import CommandT, ReturnT
from ._process import Process

__all__ = (
    "ExtraFileMethodsMixin",
    "FileMethodsMixin",
    "ProcessMethodsMixin",
    "LockingMethodsMixin",
    "TransferMethodsMixin",
    "PersistenceMethodsMixin",
)


class ExtraFileMethodsMixin(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def find(
        self,
        path: str,
        *,
        recursive: bool = False,
        hidden: bool = False,
        no_path: bool = False,
        query: str = "",
    ) -> t.Optional[t.List[str]]:
        pass

    @abc.abstractmethod
    def stat(
        self, path: str, *, dir_fd: t.Optional = None, follow_symlinks: bool = True
    ) -> t.Optional[os.stat_result]:
        pass


class FileMethodsMixin(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def delete(self, path: str, recursive: bool = False) -> str:
        pass

    @abc.abstractmethod
    def executable_exists(self, executable: str) -> bool:
        pass

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def is_file(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def is_dir(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def move(self, path_from: str, path_to: str, overwrite: bool = True) -> bool:
        pass

    @abc.abstractmethod
    def copy(
        self, path_from: str, path_to: str, recursive: bool = True, overwrite: bool = True
    ) -> bool:
        pass

    @abc.abstractmethod
    def mkdir(self, path: str, parents: bool = True) -> bool:
        pass

    @abc.abstractmethod
    def access(self, path: str, mode: str = "r") -> bool:
        pass

    @property
    @abc.abstractmethod
    def working_directory(self) -> str:
        pass


class ProcessMethodsMixin(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def spawn(
        self,
        path: str,
        args: t.Union[t.List[str], t.Dict[str, t.Any]],
        slaves: t.List[Process] = [],
        *,
        details: t.Optional[t.Dict] = None,
    ) -> t.Tuple[Process, ReturnT]:
        pass

    @abc.abstractmethod
    def start(self, *processes: Process, **kwargs) -> bool:
        pass

    @abc.abstractmethod
    def stop(self, *processes: Process, **kwargs) -> bool:
        pass

    @abc.abstractmethod
    def kill(self, *processes: Process, **kwargs) -> bool:
        pass

    @abc.abstractmethod
    def wait(self, *processes: Process, **kwargs) -> bool:
        pass


class LockingMethodsMixin(metaclass=abc.ABCMeta):
    @property
    def can_execute(self) -> bool:
        return self.is_locked

    @property
    @abc.abstractmethod
    def lock_file(self):
        pass

    @property
    @abc.abstractmethod
    def is_locked(self) -> bool:
        pass

    @abc.abstractmethod
    def lock(self) -> None:
        pass

    @abc.abstractmethod
    def unlock(self) -> bool:
        pass


class TransferMethodsMixin(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fetch(
        path_from: t.Union[Path, str], path_to: t.Union[Path, str], exclude: t.List[str]
    ) -> bool:
        pass

    @abc.abstractmethod
    def send(
        self, path_from: t.Union[Path, str], path_to: t.Union[Path, str], exclude: t.List[str]
    ) -> bool:
        pass


class PersistenceMethodsMixin(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def persistent(
        self,
        cmd: CommandT,
        chain: t.Optional[CommandT] = None,
        sudo: bool = True,
        **kwargs: t.Any,
    ) -> ReturnT:
        pass
