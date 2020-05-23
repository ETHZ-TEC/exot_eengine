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
from __future__ import annotations

import typing as t

__all__ = ("Process",)


class Process:
    def __init__(
        self,
        driver: object,
        invocation: object,
        identity: t.Union[int, str],
        slaves: t.List[Process] = [],
        duration: t.Optional[float] = None,
    ):
        """Creates a Process instance

        Args:
            driver (object): The driver
            invocation (object): The command invocation (invoke_result)
            identity (t.Union[int, str]): The identity, for example the PID or Component name
            slaves (t.List[Process], optional): The slave processes
        """
        self.driver = driver
        self.invocation = invocation
        self.identity = identity
        self.slaves = slaves
        self.duration = duration

        self.update()

    def __repr__(self) -> str:
        return "<Process {} of {!r} at {}>".format(self.identity, self.driver, hex(id(self)))

    def __hash__(self) -> int:
        return hash(self.identity)

    @property
    def exited(self) -> t.Optional[int]:
        """Gets the exit code of the process

        Returns:
            t.Optional[int]: The exit code or None if failed or running
        """
        self.update()
        return self.invocation.exited

    @property
    def stderr(self) -> t.Optional[str]:
        """Gets the invocation's standard error

        Returns:
            t.Optional[str]: The standard error
        """
        self.update()
        return self.invocation.stderr

    @property
    def stdout(self) -> t.Optional[str]:
        """Gets the invocation's standard output

        Returns:
            t.Optional[str]: The standard output
        """
        self.update()
        return self.invocation.stdout

    @property
    def children(self) -> t.List[t.Union[int, str]]:
        """Gets the identities of child properties

        Returns:
            t.List[t.Union[int, str]]: The children identities
        """
        if hasattr(self.invocation, "children"):
            return self.invocation.children or []
        else:
            return []

    @property
    def slaves(self) -> t.List[Process]:
        """Gets the slave processes

        Returns:
            t.List[Process]: The slave processes
        """
        return self._slaves

    @slaves.setter
    def slaves(self, value: t.List[Process]) -> None:
        """Sets the slave processes

        Args:
            value (t.List[Process]): The list of slave processes

        Raises:
            TypeError: Wrong type(s) supplied
        """
        if not isinstance(value, t.List):
            raise TypeError()
        if any(not isinstance(_, Process) for _ in value):
            raise TypeError()

        self._slaves = value

    def update(self):
        """Updates the process status
        """
        if hasattr(self.invocation, "update"):
            self.invocation.update()
