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
"""Concrete Unix drivers"""

import typing as t

from exot.exceptions import *

from . import extensions
from ._backend import Backend
from ._driver import Driver
from ._process import Process
from .backend_ssh import SSHBackend

__all__ = ("SSHUnixDriver",)


class SSHUnixDriver(
    extensions.rsync.TransferMethods,
    extensions.sysfs.StateMethods,
    extensions.unix.FanMethods,
    extensions.unix.LatencyMethods,
    extensions.unix.LockingMethods,
    extensions.unix.ProcessMethods,
    extensions.unix.PersistenceMethods,
    extensions.unix.ExtraFileMethods,
    extensions.unix.FileMethods,
    Driver,
    backend=SSHBackend,
):
    @property
    def lock_file(self):
        return "/var/lock/exot.lock"

    @property
    def setable_states(self):
        return ["governors", "frequencies", "latency", "fan"]

    @property
    def getable_states(self):
        return ["governors", "frequencies", "latency", "fan"]

    def spawn(
        self,
        path: str,
        args: t.List[str],
        slaves: t.List[Process] = [],
        *,
        details: t.Optional[t.Dict] = None,
    ) -> t.Tuple[Process, Backend.ReturnT]:
        if not isinstance(args, t.List):
            raise TypeError("'args' should be a list")
        if not self.executable_exists(path):
            raise RemoteCommandFailedError(f"{path!r} not executable")

        _command = [path, *args]
        _output = self.persistent(_command, chain=None, sudo=True)
        return Process(self, _output, _output.pid, slaves), _output

    def cleanup(self) -> None:
        _ = self.original_state.copy()
        # set frequencies to None for cores which had governors other than "userspace"
        _freqs = (
            [
                _["frequencies"][i] if _["governors"][i] == "userspace" else None
                for i in range(len(_["frequencies"]))
            ]
            if _["frequencies"] is not None
            else None
        )
        _["frequencies"] = _freqs
        _["latency"] = None  # disable DMA latency setter
        self.setstate(**_)  # restore state

        # Delete all nohup persistence files, even if commands have not exited successfully.
        # Their statuses will still be available in the history.
        for idx, command in enumerate(self.backend.history):
            if hasattr(command, "persistent"):
                command.update()
                self.backend._history[idx] = command
                command.cleanup()
