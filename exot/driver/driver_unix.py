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
