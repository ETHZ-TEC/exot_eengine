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
