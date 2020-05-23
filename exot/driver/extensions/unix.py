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
"""Concrete mixins for extended functionality"""

import itertools
import os
import typing as t
from shlex import quote

from exot.exceptions import *
from exot.util.logging import get_root_logger
from exot.util.misc import list2cmdline, random_string, timestamp

from .._backend import CommandT, ReturnT
from .._mixins import *
from .._process import Process


class PersistenceMethods(PersistenceMethodsMixin):
    def persistent(
        self,
        cmd: CommandT,
        chain: t.Optional[CommandT] = None,
        sudo: bool = True,
        **kwargs: t.Any,
    ) -> ReturnT:
        """Run a persistent command that is detached from the calling process

        In addition to the base ReturnT, the output contains attributes:
        -   pid: the pid of the detached process
        -   pgid: the process group id of the detached process
        -   children: child processes spawned by the shell

        Args:
            cmd (CommandT): the command to run persistently
            chain (t.Optional[CommandT], optional):
                a command to run once the primary command finishes
            sudo (bool): run commands with elevated privileges
            **kwargs (t.Any): optional keyword arguments: "history", "command_only"
        """
        if not isinstance(cmd, (str, t.List)):
            raise TypeError(f"'cmd' ({type(cmd)}) not str or list")
        if isinstance(cmd, t.List):
            cmd = list2cmdline(cmd)

        if chain and not isinstance(cmd, (str, t.List)):
            raise TypeError(f"'chain' ({type(chain)}) not str or list")
        if chain and isinstance(cmd, t.List):
            chain = list2cmdline(chain)

        if "history" in kwargs:
            history = kwargs.pop("history")
            if not isinstance(history, bool):
                _ = f"keyword argument 'history' should be bool, got: {type(history)}"
                raise TypeError(_)
        else:
            history = True

        # nohup temporary files
        nohup_id = timestamp() + "_" + random_string(5)
        nohup = {
            "pre": "/usr/bin/env bash -c",
            "id": nohup_id,
            "out": ".nohup-" + nohup_id + "-out",
            "err": ".nohup-" + nohup_id + "-err",
            "ret": ".nohup-" + nohup_id + "-ret",
            "rch": ".nohup-" + nohup_id + "-rch",
        }

        nohup["files"] = [nohup["out"], nohup["err"], nohup["ret"], nohup["rch"]]

        # echo the exit code to a unique temporary file
        nohup["wrap"] = quote(
            cmd
            + "; echo $? > {}{}".format(
                nohup["ret"], f"; {chain}; echo $? > {nohup['rch']}" if chain else ""
            )
        )
        # complete nohup + disown command
        _cmd = "nohup {pre} {wrap} 1>{out} 2>{err} & disown && echo $!".format(**nohup)

        if kwargs.get("command_only"):
            return _cmd

        invocation = self.command(_cmd, sudo=sudo, history=False)

        def _read():
            sep = random_string(5)
            line = " <(echo {}) ".format(sep)
            command = "cat " + line.join(nohup["files"])
            result = self.command(command)
            values = [_.strip("\n") for _ in result.stdout.split(sep)]
            assert len(values) == 4, "expected exactly 4 values after splitting"

            # return codes
            values[2] = int(values[2]) if values[2] else None
            values[3] = int(values[3]) if values[3] else None

            # for stderr, ignore the first line with contains "nohup: ignoring input"
            values[0] = values[0] if values[2] is not None else None
            values[1] = "\n".join(values[1].split("\n")[1:]) if values[2] is not None else None

            return dict(zip(["out", "err", "ret", "rch"], values))

        def _cleanup():
            self.command(["rm", *nohup["files"]])

        nohup["nohup_pid"] = int(invocation.stdout.strip())
        nohup["pre_pid"] = int(self.command(f"pgrep -P {nohup['nohup_pid']}").stdout.strip())

        nohup["pid"] = nohup["pre_pid"]
        _pgid = self.command(f"ps -o pgid= {nohup['pre_pid']}")
        _ppid = self.command(f"ps -o pgid= {nohup['pre_pid']}")
        _cpids = self.command(f"pgrep -P {nohup['pre_pid']}")

        nohup["ppid"] = int(_ppid.stdout.strip()) if _ppid else None
        nohup["pgid"] = int(_pgid.stdout.strip()) if _pgid else None
        nohup["children"] = [int(_) for _ in _cpids.stdout.split("\n")] if _cpids else None
        nohup["runner"] = self.command
        nohup["command"] = _cmd
        nohup["chain"] = chain
        nohup["chain_exited"] = None

        nohup["read"] = _read
        nohup["cleanup"] = _cleanup
        nohup["cleaned_up"] = False

        invocation.pid = nohup["pid"]
        invocation.pgid = nohup["pgid"]
        invocation.children = nohup["children"]
        invocation.command = cmd
        invocation.persistent = nohup

        def update():
            assert hasattr(invocation, "persistent")
            values = invocation.persistent["read"]()

            if not invocation.persistent["cleaned_up"]:
                invocation.stdout = values["out"]
                invocation.stderr = values["err"]
                invocation.exited = values["ret"]
                invocation.persistent["chain_exited"] = values["rch"]

            if invocation.exited == 0 and not invocation.persistent["chain"]:
                invocation.persistent["cleanup"]()
                invocation.persistent["cleaned_up"] = True
            elif (
                invocation.exited == 0
                and invocation.persistent["chain"]
                and invocation.persistent["chain_exited"] == 0
            ):
                invocation.persistent["cleanup"]()
                invocation.persistent["cleaned_up"] = True

        invocation.cleanup = _cleanup
        invocation.update = update
        invocation.update()

        if history and hasattr(self.backend, "_history"):
            self.backend._history.append(invocation)

        return invocation


class ExtraFileMethods(ExtraFileMethodsMixin):
    def stat(
        self, path: str, *, dir_fd: t.Optional = None, follow_symlinks: bool = True
    ) -> t.Optional[os.stat_result]:
        # Linux stat accepts the following format sequences for the minimum required
        # set of values needed for os.stat_result:
        #   st_mode = %f,   st_ino = %i,    st_dev = %d,   st_nlink = %h
        #   st_uid = %u,    st_gid = %g,    st_size = %s,  st_atime = %X
        #   st_mtime = %Y,  st_ctime = %Z
        _format = r"%f %i %d %h %u %g %s %X %Y %Z"
        _output = self.command(["stat", "-L" if follow_symlinks else None, "-c", _format, path])

        if not _output.ok:
            return None

        _vals = _output.stdout.split()
        assert len(_vals) == 10, "the formatted 'stat' output should be 10 elements"
        _vals[0] = int(_vals[0], base=16)  # raw access mode is in hex
        _stat_result = tuple(int(v) for v in _vals)

        return os.stat_result(_stat_result)

    def find(
        self,
        path: str,
        *,
        recursive: bool = False,
        hidden: bool = False,
        no_path: bool = False,
        query: str = "",
    ) -> t.Optional[t.List[str]]:
        _command = "find {p} {n} {h} {r} {q}".format(
            p=path,
            n=f"-not -path {path}" if no_path else None,
            r=None if recursive else "-maxdepth 1",
            q=query,
            h=r"-not -regex '.*/\..*'" if hidden else None,
        )
        _output = self.command(_command)
        if _output.ok:
            return _output.stdout.splitlines()
        else:
            return None


class LockingMethods(FileMethodsMixin, LockingMethodsMixin):
    @property
    def is_locked(self) -> bool:
        return self.exists(self.lock_file)

    def lock(self) -> None:
        self.command(["echo", os.uname().nodename, " | ", "$(date)", ">>", self.lock_file])

    def unlock(self) -> bool:
        return self.delete(self.lock_file)


class FileMethods(FileMethodsMixin):
    def delete(self, path: str, recursive: bool = False) -> str:
        return self.command(["rm", "-fr" if recursive else None, path]).ok

    def exists(self, path: str) -> bool:
        return self.command(["test", "-e", path]).ok

    def is_file(self, path: str) -> bool:
        return self.command(["test", "-f", path]).ok

    def is_dir(self, path: str) -> bool:
        return self.command(["test", "-d", path]).ok

    def move(self, path_from: str, path_to: str, overwrite: bool = True) -> bool:
        return self.command(["mv", "-v", None if overwrite else "-n", path_from, path_to]).ok

    def copy(
        self, path_from: str, path_to: str, recursive: bool = True, overwrite: bool = True
    ) -> bool:
        return self.command(
            [
                "cp",
                "-v",
                None if overwrite else "-n",
                "-r" if recursive else None,
                path_from,
                path_to,
            ]
        ).ok

    def mkdir(self, path: str, parents: bool = True) -> bool:
        return self.command(["mkdir", "-vp" if parents else None, path]).ok

    def access(self, path: str, mode: str = "r") -> bool:
        if mode not in ["r", "w", "x"]:
            raise ValueError("'mode' argument must be 'r', 'w', or 'x', got: " + str(mode))
        return self.command(["test", "-" + mode, path]).ok

    def executable_exists(self, executable: str) -> bool:
        """Checks if an executable exists locally or on the system

        Args:
            executable (str): The executable, either a path or an app name

        Returns:
            bool: True if exists and is an executable
        """
        assert isinstance(executable, str)

        _exists_local = self.exists(executable)
        _access_local = self.access(executable, "x")
        _local_valid = _exists_local and _access_local

        if not _local_valid:
            _exists_system = self.command(["which", executable], sudo=True)
            return _exists_system.ok
        else:
            return _local_valid

    @property
    def working_directory(self) -> str:
        return self.command(["pwd"]).stdout.strip()


class LatencyMethods:
    @property
    def latency(self) -> t.Optional[int]:
        """Get the CPU DMA latency from devfs

        Returns:
            t.Optional[int]: an integer value or None if unsuccessful
        """
        _output = self.command(["cat", "/dev/cpu_dma_latency"], sudo=True)

        if _output.ok:
            try:
                return int.from_bytes(_output.stdout.strip().encode(), "little")
            except Exception:
                return None
        else:
            return None

    @latency.setter
    def latency(self, value: t.Optional[int]) -> None:
        existing_pgid = getattr(self, "_latency_setter_pgid", None)

        if existing_pgid:
            # must kill whole process group, the pgid of children processes
            # should be the same as calling parent
            if not self.unix_kill(existing_pgid, pgid=True):
                _ = self.backend.history[-1]
                get_root_logger().warning(
                    f"failed to kill CPU DMA setter, exit code {_.exited}, "
                    f"stderr {_.stderr!r}"
                )
            else:
                delattr(self, "_latency_setter_pgid")

        # setting to None will just kill the existing dma latency setter
        if value is None:
            return

        elif not isinstance(value, int):
            raise TypeError("latency accepts integer values")

        # Accepts bit-endian values as byte strings or hex values (without 0x prefix)
        value = value.to_bytes(4, "big").hex()

        # CPU DMA latency will be set as long as a file descriptor is held, the value
        # has to be written only once.
        inner = "exec 4> /dev/cpu_dma_latency; echo -ne {} >&4; sleep 99999".format(str(value))
        outer = ["sudo", "/usr/bin/env", "bash", "-c", inner]

        _ = self.persistent(outer)
        self._latency_setter_pgid = _.pgid


class FanMethods:
    @property
    def fan_path(self) -> t.Optional[t.Dict]:
        """Gets the path to fan settings on the device

        Returns:
            t.Optional[t.Dict]: The {key, path, exists} of an existing fan path,
                                   None if no path has been found.
        """
        # The keys are only used to identify various endpoints
        _ = {
            "thinkpad": "/proc/acpi/ibm/fan",
            "odroid-xu3": "/sys/devices/odroid_fan.14/",
            "odroid-2": "/sys/devices/platform/odroidu2-fan/",
            "odroid-xu4": "/sys/devices/platform/pwm-fan/hwmon/hwmon0/",
            "pwm-fan": "/sys/bus/platform/devices/pwm-fan/",
        }

        availability = [{"key": k, "path": v, "exists": self.exists(v)} for k, v in _.items()]
        existing_paths = [p for p in availability if p["exists"]]
        assert len(existing_paths) <= 1, "There should be at most 1 fan settings path"
        return existing_paths[0] if existing_paths else None

    @property
    def fan(self) -> t.Optional[t.Union[str, tuple]]:
        """Gets fan settings

        Returns:
            t.Optional[t.Union[str, tuple]]: The fan settings
        """
        fan = self.fan_path

        if not fan:
            get_root_logger().warning("setting fan failed due to no fan path found")
            return None

        if fan["key"] == "thinkpad":
            _ = self.backend.run(["cat", fan["path"], "|", "grep", "'^level'"])
            return _.stdout.split(":")[1].strip() if _.ok else None

        elif fan["key"] == "odroid-xu3" or fan["key"] == "odroid-2":
            _ = self.backend.run(["cat", fan["path"] + "{fan_mode,pwm_duty}"])
            return tuple(_.stdout.splitlines()) if _.ok else None

        elif fan["key"] == "odroid-xu4":
            _ = self.backend.run(["cat", fan["path"] + "{automatic,pwm1}"])
            return tuple(_.stdout.splitlines()) if _.ok else None

        elif fan["key"] == "pwm-fan":
            _ = self.backend.run(["cat", fan["path"] + "cur_pwm"])
            return _.stdout.strip() if _.ok else None

    @fan.setter
    def fan(self, value: t.Union[bool, str, int, tuple]) -> None:
        """Sets fan settings

        Args:
            value (t.Union[str, int, tuple]): Value appropriate for the specific fan endpoints

        Returns:
            None: Returns early if no fan path is available

        Raises:
            DriverTypeError: Wrong type supplied to the specific fan endpoint
            DriverValueError: Wrong value supplied to the specific fan endpoint
        """
        fan = self.fan_path

        if not fan:
            get_root_logger().warning("setting fan failed due to no fan path found")
            return

        if fan["key"] == "thinkpad":
            if isinstance(value, bool):
                value = "7" if value is True else "auto"
            elif not isinstance(value, str):
                raise DriverTypeError("thinkpad fan setting accepts a bool or a single str")

            _ = self.backend.run(
                ["echo", "level", quote(value), "|", "sudo", "tee", fan["path"]]
            )

            if not _.ok:
                get_root_logger().warning(
                    f"setting thinkpad fan to {value!r} failed, exit code: {_.exited}, "
                    f"stderr: {_.stderr}"
                )

        elif fan["key"] == "odroid-xu3" or fan["key"] == "odroid-2":
            if isinstance(value, bool):
                value = ("1", "255") if value is True else ("0", "")
            elif not isinstance(value, (tuple, list)) and len(value) != 2:
                raise DriverTypeError("odroid fan setting accept a bool or a 2-tuple")
            assert all(isinstance(v, str) for v in value), "values must be strings"

            _ = self.backend.run(
                f"echo {quote(value[0])} | sudo tee {fan['path'] + 'fan_mode'}"
            )

            if not _.ok:
                get_root_logger().warning(
                    f"setting odroid fan MODE to {value[0]!r} failed, "
                    f"exit code: {_.exited}, stderr: {_.stderr}"
                )

            if value[1]:
                _pwm = self.backend.run(
                    f"echo {quote(value[1])} | sudo tee {fan['path'] + 'pwm_duty'}"
                )

                if not _pwm.ok:
                    get_root_logger().warning(
                        f"setting odroid fan PWM to {value[1]!r} failed, "
                        f"exit code: {_pwm.exited}, stderr: {_pwm.stderr}"
                    )

        elif fan["key"] == "odroid-xu4":
            if isinstance(value, bool):
                value = ("0", "255") if value is True else ("1", "0")
            elif not isinstance(value, (tuple, list)) and len(value) != 2:
                raise DriverTypeError("odroid fan setting accept a bool or a 2-tuple")
            assert all(isinstance(v, str) for v in value), "values must be strings"

            _ = self.backend.run(
                f"echo {quote(value[0])} | sudo tee {fan['path'] + 'automatic'}"
            )

            if not _.ok:
                get_root_logger().warning(
                    f"setting odroid fan 'AUTOMATIC' to {value[0]!r} failed, "
                    f"exit code: {_.exited}, stderr: {_.stderr}"
                )

            _ = self.backend.run(f"echo {quote(value[1])} | sudo tee {fan['path'] + 'pwm1'}")

            if not _.ok:
                get_root_logger().warning(
                    f"setting odroid fan 'PWM1' to {value[1]!r} failed, "
                    f"exit code: {_.exited}, stderr: {_.stderr}"
                )

        elif fan["key"] == "pwm-fan":
            if isinstance(value, bool):
                value = 255 if value is True else 0
            elif isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    raise DriverValueError(
                        f"could not convert provided value to integer: {value}"
                    )
            elif isinstance(value, int):
                pass
            else:
                raise DriverTypeError(f"pwm-fan accepts bool, int, and str; got: {type(value)}")

            if value > 255 or value < 0:
                raise DriverValueError(
                    f"pwm-fan accepts values in range [0, 255], got: {value}"
                )

            value = str(value)

            _ = self.backend.run(f"echo {quote(value)} | sudo tee {fan['path'] + 'target_pwm'}")

            if not _.ok:
                get_root_logger().warning(
                    f"setting pwm-fan to value {value} failed, "
                    f"exit code: {_.exited}, stderr: {_.stderr}"
                )


# fmt: off
VALID_SIGNALS = {
     1: "HUP",   2: "INT",     3: "QUIT",   4: "ILL",      5: "TRAP",   6: "ABRT",
     7: "BUS",   8: "FPE",     9: "KILL",  10: "USR1",    11: "SEGV",  12: "USR2",
    13: "PIPE", 14: "ALRM",   15: "TERM",  16: "STKFLT",  17: "CHLD",  18: "CONT",
    19: "STOP", 20: "TSTP",   21: "TTIN",  22: "TTOU",    23: "URG",   24: "XCPU",
    25: "XFSZ", 26: "VTALRM", 27: "PROF",  28: "WINCH",   29: "POLL",  30: "PWR",
    31: "SYS",
}
# fmt: on


class ProcessMethods(ProcessMethodsMixin):
    def unix_signal(self, *pids: int, sig: t.Union[str, int]) -> bool:
        if isinstance(sig, str):
            assert sig in VALID_SIGNALS.values(), "signal must be a valid Unix signal"
        elif isinstance(sig, int):
            assert sig in VALID_SIGNALS.keys(), "signal must be in the range [1, 31]"
            sig = VALID_SIGNALS[sig]
        assert len(pids) >= 1, "no pids found!"
        assert all(isinstance(_, (str, int)) for _ in pids), "each pid must be int or str"

        _output = self.command(["kill", "-s", sig, *[str(_) for _ in pids]], sudo=True)
        return _output.ok

    def unix_pkill(self, *pids: str, flag: str = "", sig: t.Union[str, int], **kwargs) -> bool:
        if isinstance(sig, str):
            assert sig in VALID_SIGNALS.values(), "signal must be a valid Unix signal"
        elif isinstance(sig, int):
            assert sig in VALID_SIGNALS.keys(), "signal must be in the range [1, 31]"
            sig = VALID_SIGNALS[sig]
        assert len(pids) >= 1, "no pids found!"
        pids = [_ for _ in pids if _ is not None]
        assert all(isinstance(_, (str, int)) for _ in pids), "each pid must be int or str"
        assert len(kwargs) <= 1, "at most 1 keyword argument is accepted"

        flag = None

        if kwargs.pop("pgid", False):
            flag = "-g"
        elif kwargs.pop("ppid", False):
            flag = "-P"
        elif kwargs.pop("sid", False):
            flag = "-s"
        elif "flag" in kwargs:
            flag = kwargs.pop("flag")
            assert isinstance(flag, str), "flag must be a str"
            assert flag[0] == "-", "flag must have a preceding '-'"

        args = [[flag, str(pid)] for pid in pids] if flag else [[str(pid)] for pid in pids]

        _command = "; ".join(list2cmdline(["pkill", "--signal", sig, *_]) for _ in args)
        _output = self.command(_command, sudo=True)
        return _output.ok

    def unix_kill(self, *pids: int, **kwargs) -> bool:
        if not kwargs:
            return self.unix_signal(*pids, sig="KILL")
        else:
            return self.unix_pkill(*pids, sig="KILL", **kwargs)

    def unix_stop(self, *pids: int, **kwargs) -> bool:
        if not kwargs:
            return self.unix_signal(*pids, sig="INT")
        else:
            return self.unix_pkill(*pids, sig="INT", **kwargs)

    def unix_start(self, *pids: int, **kwargs) -> bool:
        if not kwargs:
            return self.unix_signal(*pids, sig="USR1")
        else:
            return self.unix_pkill(*pids, sig="USR1", **kwargs)

    def unix_wait(self, *pids: t.Union[int, ReturnT], refresh_period: float = 5.0) -> bool:
        assert len(pids) >= 1, "no pids found!"
        assert isinstance(refresh_period, (int, float)), "refresh_period must be a number"

        if all(isinstance(pid, ReturnT) for pid in pids):
            assert all(
                hasattr(pid, "persistent") for pid in pids
            ), "all must be persistent command returns"

            _nohup_rets = [pid.persistent["ret"] for pid in pids]
            _cat = " ".join(_nohup_rets)
            _command = "while sleep {}; do cat {} && break; done".format(_sleep_duration, _cat)

            self.command(_command)
            return True
        elif all(isinstance(_, (str, int)) for _ in pids):
            pids = [str(_) for _ in pids]
            _sleep_duration = "{:.1f}".format(refresh_period)

            # chech if 'ps -p' fails correctly
            ps_p_expect_ok = self.command("ps -p 1").ok is True
            ps_p_expect_fail = self.command("ps -p abcd").ok is False
            ps_p_ok = ps_p_expect_ok and ps_p_expect_fail

            if not ps_p_ok:
                # chech if 'test -e /proc' fails correctly
                test_e_proc_expect_ok = self.command("test -e /proc/1").ok is True
                test_e_proc_expect_fail = self.command("test -e /proc/abcd").ok is False
                test_e_proc_ok = test_e_proc_expect_ok and test_e_proc_expect_fail

                if test_e_proc_ok:
                    # the '-o' flag logical-or's the arguments to test, if any of the process
                    # directories still exist, keep waiting.
                    _flags = " -o -e /proc/".join([str(pid) for pid in pids])
                    _cmd = "test -e /proc/{}".format(_flags)
                else:
                    # none of the methods work...
                    raise RemoteCommandFailedError(
                        "Waiting with neither 'ps -p ...' nor 'test -e /proc/...' works on "
                        "the platform"
                    )
            else:
                _flags = "-p ".join([str(pid) for pid in pids])
                _cmd = "ps -p {}".format(_flags)

            _command = "while sleep {}; do {} || break; done".format(_sleep_duration, _cmd)

            _ = self.command(_command)
            return _.ok
        else:
            raise RemoteCommandFailedError(
                "Must supply either pids as integers or strings, or backend persistent commands"
            )

    @staticmethod
    def _get_pids(*processes: Process) -> t.List[int]:
        return [process.invocation.pid for process in processes]

    @staticmethod
    def _get_cpids(*processes: Process) -> t.List[int]:
        _ = [process.invocation.children for process in processes]
        return list(itertools.chain.from_iterable(_))

    @staticmethod
    def _get_pgids(*processes: Process) -> t.List[int]:
        return [process.invocation.pgid for process in processes]

    def start(self, *processes: Process, **kwargs) -> bool:
        pids = self._get_pids(*processes)
        cpids = self._get_cpids(*processes)

        return (
            self.unix_start(*cpids, **kwargs)
            if cpids
            else (self.unix_start(*pids, **kwargs) if pids else False)
        )

    def stop(self, *processes: Process, **kwargs) -> bool:
        pids = self._get_pids(*processes)
        cpids = self._get_cpids(*processes)
        pgids = self._get_pgids(*processes)

        if cpids:
            return self.unix_stop(*cpids, **kwargs)
        elif pgids:
            return self.unix_stop(*pgids, **kwargs)
        else:
            return self.unix_stop(*pids, **kwargs)

    def kill(self, *processes: Process, **kwargs) -> bool:
        pids = self._get_pids(*processes)
        cpids = self._get_cpids(*processes)
        pgids = self._get_pgids(*processes)

        if cpids:
            return self.unix_kill(*cpids, **kwargs)
        elif pgids:
            return self.unix_kill(*pgids, **kwargs)
        else:
            return self.unix_kill(*pids, **kwargs)

    def wait(self, *processes: Process, **kwargs) -> bool:
        pids = self._get_pids(*processes)

        return self.unix_wait(*pids, **kwargs) if pids else False
