"""Abstract mixin classes to separate concerns and define interfaces"""

import abc
import os
import typing as t
from pathlib import Path

from ._backend import ReturnT

__all__ = (
    "FileMethodsMixin",
    "ProcessMethodsMixin",
    "SetupMethodsMixin",
    "TransferMethodsMixin",
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


class ProcessMethodsMixin(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def spawn(self, path: str, args: t.List[str]) -> t.Tuple[int, ReturnT]:
        pass

    @abc.abstractmethod
    def signal(self, *pids: int, sig: t.Union[str, int]) -> bool:
        pass

    @abc.abstractmethod
    def start(self, *pids: int, **kwargs) -> bool:
        pass

    @abc.abstractmethod
    def stop(self, *pids: int, **kwargs) -> bool:
        pass

    @abc.abstractmethod
    def kill(self, *pids: int, **kwargs) -> bool:
        pass

    @abc.abstractmethod
    def wait(self, *pids: int, **kwargs) -> bool:
        pass


class SetupMethodsMixin(metaclass=abc.ABCMeta):
    @property
    def can_execute(self) -> bool:
        return self.is_locked

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

    @abc.abstractmethod
    def getstate(self) -> t.Dict:
        pass

    @abc.abstractmethod
    def setstate(self, value: t.Dict):
        pass

    @abc.abstractmethod
    def cleanup(self):
        pass


class TransferMethodsMixin(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fetch(self, path_from: Path, path_to: Path) -> bool:
        pass

    @abc.abstractmethod
    def send(self, path_from: Path, path_to: Path) -> bool:
        pass
