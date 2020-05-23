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
"""Driver backend base class"""

from __future__ import annotations

import abc
import typing as t
from contextlib import AbstractContextManager

import invoke

from exot.exceptions import *
from exot.util.mixins import Configurable

__all__ = ("Backend", "result_to_dict", "result_to_print_ready_dict")

"""
Backend
-------

Synopsis & signatures::

can_connect           (self) -> 'bool'
connect               (self) -> 'None'
disconnect            (self) -> 'None'
instantiate           (*args, **kwargs) -> 'object'
required_config_keys  () -> List[str]
run                   (self, cmd: 'CommandT', **kwargs) -> 'ReturnT'
sudo                  (self, cmd: 'CommandT', **kwargs) -> 'ReturnT'
validate              (self) -> NoReturn
"""


CommandT = t.Union[t.List[str], str]
ReturnT = invoke.runners.Result

"""
invoke.runners.Result is used as a wrapper for shell/connection output. It contains
parameters such as command, stdout, stderr, return_code, ok, failed.

To create a Result from simple command output, one can call:

>>> r = invoke.runners.Result(command="my_command", stdout="hello, world!", exited=1)
>>> r.command
'my_command'
>>> r.stdout
'hello, world!'
>>> r.stderr
''
>>> r.exited == r.return_code == 1
True
>>> r.ok
False
"""


class Backend(Configurable, AbstractContextManager, configure="backend", metaclass=abc.ABCMeta):

    """The base class for driver backends

    Attributes:
        CommandT (TYPE): The command type passed to the backend
        ReturnT (TYPE): The type returned by the backend when a command completes
    """

    CommandT = CommandT
    ReturnT = ReturnT

    def __init__(self, *args, **kwargs) -> None:
        """Initialises the backend

        Args:
            *args: Positional arguments passed to the base class
            **kwargs: Keyword arguments passed to the base class
        """
        super().__init__(*args, **kwargs)

        # Command/Result history
        # ReturnT should contain the command in the first place, therefore logging
        # commands separately is not required.
        self._history = []

    def __enter__(self) -> object:
        """Creates a backend object to be used as a context manager

        Returns:
            object: The backend, connected if possible
        """
        if not self.connected:
            if self.can_connect():
                self.connect()
        return self

    def __exit__(self, *exc) -> None:
        """Destroys the backend object after exiting a 'with' context

        Args:
            *exc: The exceptions array
        """
        self.disconnect()

    def __repr__(self) -> str:
        """Represents the backend instance as a string

        Returns:
            str: The string representation
        """
        return "<{}.{} at {} ({}, {})>".format(
            self.__module__,
            self.__class__.__name__,
            hex(id(self)),
            "configured" if self.configured else "not configured",
            "active" if self.connected else "inactive",
        )

    @classmethod
    def instantiate(cls, *args, **kwargs) -> object:
        """Get a connected Backend instance"""
        instance = cls(*args, **kwargs)
        if not instance.can_connect():
            raise CouldNotConnectError("cannot connect")
        instance.connect()
        return instance

    @abc.abstractmethod
    def can_connect(self) -> bool:
        """Can the backend connect to its target?

        Returns:
            bool: True if properly configured
        """
        return self.configured

    @property
    @abc.abstractmethod
    def connected(self) -> bool:
        """Is the backend connected?
        """
        pass

    @property
    @abc.abstractmethod
    def connection(self) -> object:
        """Gets the underlying implementation-specific connection object
        """
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        """Connects the backend
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnects the backend
        """
        pass

    @abc.abstractmethod
    def run(self, cmd: CommandT, **kwargs) -> ReturnT:
        """Runs a command on the target using the backend

        Args:
            cmd (CommandT): The command
            **kwargs: Optional keyword arguments
        """
        pass

    def sudo(self, cmd: CommandT, **kwargs) -> ReturnT:
        """Runs a command on the target using the backend with elevated privileges

        Args:
            cmd (CommandT): The command
            **kwargs: Optional keyword arguments
        """
        return self.run(cmd, **kwargs)

    @property
    def history(self) -> t.List[ReturnT]:
        """Get the command/result history"""
        return self._history


def result_to_dict(result: ReturnT) -> dict:
    """Produces a dict from a ReturnT"""
    assert isinstance(result, ReturnT), "can only process backend invocation results"
    _ = {}
    for attribute in ["command", "encoding", "exited", "stdout", "stderr", "persistent", "pid"]:
        if hasattr(result, attribute):
            _[attribute] = getattr(result, attribute)
    return _


def result_to_print_ready_dict(result: ReturnT) -> dict:
    """Produce a dict from a ReturnT with some details omitted"""
    assert isinstance(result, ReturnT), "can only format backend invocation results"
    result = result_to_dict(result)
    if "persistent" in result:
        _ = result["persistent"]
        result["persistent"] = {
            "id": _["id"],
            "pid": _["pid"],
            "pgid": _["pgid"],
            "command": _["command"],
        }
    return result
