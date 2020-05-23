"""Concrete Unix drivers"""

import os
import typing as t
from pathlib import Path
from shlex import quote

from exot.exceptions import *
from exot.util.file import _path_type_check, check_access
from exot.util.logging import get_root_logger
from exot.util.misc import list2cmdline

from ._backend import Backend
from ._driver import Driver
from ._extensions import NohupPersistenceMixin
from ._mixins import ExtraFileMethodsMixin
from .ssh import SSHBackend

__all__ = ("SSHUnixDriver",)

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

"""
SSHUnixDriver
----------------

Synopsis & signatures::

access                (self, path: str, mode: str = 'r') -> bool
available_frequencies (self, cpu: int) -> List[int]
available_governors   (self, cpu: int) -> List[str]
can_connect           (self) -> 'bool'
cleanup               (self) -> None
command               (self, cmd: 'Backend.CommandT', **kwargs: 'dict') -> 'Backend.ReturnT'
connect               (self) -> 'None'
copy                  (self, path_from: str, path_to: str, recursive: bool = True, overwrite: bool = True) -> bool
delete                (self, path: str, recursive: bool = False) -> str
disconnect            (self) -> 'None'
exists                (self, path: str) -> bool
fetch                 (self, path_from: Union[pathlib.Path, str], path_to: Union[pathlib.Path, str]) -> None
find                  (self, path: str, *, recursive: bool = False, hidden: bool = False, no_path: bool = False, query: str = '') -> Union[List[str], NoneType]
getstate              (self) -> Dict
instantiate           (*args, **kwargs) -> 'Driver'
is_dir                (self, path: str) -> bool
is_file               (self, path: str) -> bool
kill                  (self, *pids: int, **kwargs) -> bool
mkdir                 (self, path: str, parents: bool = True) -> bool
move                  (self, path_from: str, path_to: str, overwrite: bool = True) -> bool
pkill                 (self, *pids: str, flag: str = '', sig: Union[str, int], **kwargs) -> bool
send                  (self, path_from: Union[pathlib.Path, str], path_to: Union[pathlib.Path, str]) -> None
setstate              (self, **kwargs) -> None
signal                (self, *pids: int, sig: Union[str, int]) -> bool
spawn                 (self, path: str, args: List[str], sudo: bool = True) -> Tuple[int, invoke.runners.Result]
start                 (self, *pids: int, **kwargs) -> bool
stat                  (self, path: str, *, dir_fd: Optional = None, follow_symlinks: bool = True) -> Union[os.stat_result, NoneType]
stop                  (self, *pids: int, **kwargs) -> bool
wait                  (self, *pids: int, **kwargs) -> bool
"""


class SSHUnixDriver(Driver, NohupPersistenceMixin, ExtraFileMethodsMixin, backend=SSHBackend):
    # File methods
    def stat(
        self, path: str, *, dir_fd: t.Optional = None, follow_symlinks: bool = True
    ) -> t.Optional[os.stat_result]:
        # Linux stat accepts the following format sequences for the minimum required
        # set of values needed for os.stat_result:
        #   st_mode = %f,   st_ino = %i,    st_dev = %d,   st_nlink = %h
        #   st_uid = %u,    st_gid = %g,    st_size = %s,  st_atime = %X
        #   st_mtime = %Y,  st_ctime = %Z
        _format = r"%f %i %d %h %u %g %s %X %Y %Z"
        _output = self.backend.run(
            ["stat", "-L" if follow_symlinks else None, "-c", _format, path]
        )

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
        _output = self.backend.run(_command)
        if _output.ok:
            return _output.stdout.split("\n")
        else:
            return None

    @property
    def lock_file(self):
        return "/var/lock/karajan.lock"

    @property
    def is_locked(self) -> bool:
        return self.exists(self.lock_file)

    def lock(self) -> None:
        self.backend.run(["touch", self.lock_file])

    def unlock(self) -> bool:
        return self.delete(self.lock_file)

    def delete(self, path: str, recursive: bool = False) -> str:
        _output = self.backend.run(["rm", "-fr" if recursive else None, path])
        return _output.ok

    def exists(self, path: str) -> bool:
        _output = self.backend.run(["test", "-e", path])
        return _output.ok

    def is_file(self, path: str) -> bool:
        _output = self.backend.run(["test", "-f", path])
        return _output.ok

    def is_dir(self, path: str) -> bool:
        _output = self.backend.run(["test", "-d", path])
        return _output.ok

    def move(self, path_from: str, path_to: str, overwrite: bool = True) -> bool:
        _output = self.backend.run(
            ["mv", "-v", None if overwrite else "-n", path_from, path_to]
        )
        return _output.ok

    def copy(
        self, path_from: str, path_to: str, recursive: bool = True, overwrite: bool = True
    ) -> bool:
        _output = self.backend.run(
            [
                "cp",
                "-v",
                None if overwrite else "-n",
                "-a" if recursive else None,
                path_from,
                path_to,
            ]
        )
        return _output.ok

    def mkdir(self, path: str, parents: bool = True) -> bool:
        _output = self.backend.run(["mkdir", "-vp" if parents else None, path])
        return _output.ok

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
            _exists_system = self.backend.sudo(["which", executable])
            return _exists_system.ok
        else:
            return _local_valid

    def access(self, path: str, mode: str = "r") -> bool:
        assert mode in ["r", "w", "x"], "'mode' argument must be 'r', 'w', or 'x'"
        _output = self.backend.run(["test", "-" + mode, path])
        return _output.ok

    def spawn(
        self, path: str, args: t.List[str], *, slave_pids: t.List[int] = [], sudo: bool = True
    ) -> t.Tuple[int, Backend.ReturnT]:
        if not isinstance(args, t.List):
            raise TypeError("'args' should be a list")
        if not self.executable_exists(path):
            raise RemoteCommandFailedError(f"{path!r} not executable")

        _command = ["sudo" if sudo else None, path, *args]
        _chain = None

        if slave_pids:
            assert isinstance(slave_pids, t.List), "slave_pids must be a list"
            assert all(isinstance(_, int) for _ in slave_pids), "slave_pids must have int"
            _kill_list = " ".join([str(_) for _ in slave_pids])
            _chain = "sudo kill -INT {}".format(_kill_list)

        _output = self.persistent(_command, chain=_chain)
        return _output.pid, _output

    # Signals:
    #     1 HUP      2 INT      3 QUIT     4 ILL      5 TRAP     6 ABRT     7 BUS
    #     8 FPE      9 KILL    10 USR1    11 SEGV    12 USR2    13 PIPE    14 ALRM
    #    15 TERM    16 STKFLT  17 CHLD    18 CONT    19 STOP    20 TSTP    21 TTIN
    #    22 TTOU    23 URG     24 XCPU    25 XFSZ    26 VTALRM  27 PROF    28 WINCH
    #    29 POLL    30 PWR     31 SYS
    def signal(self, *pids: int, sig: t.Union[str, int]) -> bool:
        if isinstance(sig, str):
            assert sig in VALID_SIGNALS.values(), "signal must be a valid Unix signal"
        elif isinstance(sig, int):
            assert sig in VALID_SIGNALS.keys(), "signal must be in the range [1, 31]"
            sig = VALID_SIGNALS[sig]
        assert len(pids) >= 1, "at least 1 positional argument required"
        assert all(isinstance(_, (str, int)) for _ in pids), "each pid must be int or str"

        _output = self.backend.sudo(["kill", "-s", sig, *[str(_) for _ in pids]])
        return _output.ok

    def pkill(self, *pids: str, flag: str = "", sig: t.Union[str, int], **kwargs) -> bool:
        if isinstance(sig, str):
            assert sig in VALID_SIGNALS.values(), "signal must be a valid Unix signal"
        elif isinstance(sig, int):
            assert sig in VALID_SIGNALS.keys(), "signal must be in the range [1, 31]"
            sig = VALID_SIGNALS[sig]
        assert len(pids) >= 1, "at least 1 positional argument required"
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

        _command = "; ".join(list2cmdline(["sudo", "pkill", "--signal", sig, *_]) for _ in args)
        _output = self.backend.run(_command)
        return _output.ok

    def kill(self, *pids: int, **kwargs) -> bool:
        if not kwargs:
            return self.signal(*pids, sig="KILL")
        else:
            return self.pkill(*pids, sig="KILL", **kwargs)

    def stop(self, *pids: int, **kwargs) -> bool:
        if not kwargs:
            return self.signal(*pids, sig="INT")
        else:
            return self.pkill(*pids, sig="INT", **kwargs)

    def start(self, *pids: int, **kwargs) -> bool:
        if not kwargs:
            return self.signal(*pids, sig="USR1")
        else:
            return self.pkill(*pids, sig="USR1", **kwargs)

    def wait(self, *pids: t.Union[int, Backend.ReturnT], refresh_period: float = 5.0) -> bool:
        assert len(pids) >= 1, "at least 1 positional argument required"
        assert isinstance(refresh_period, (int, float)), "refresh_period must be a number"

        if all(isinstance(pid, Backend.ReturnT) for pid in pids):
            assert all(
                hasattr(pid, "persistent") for pid in pids
            ), "all must be persistent command returns"

            _nohup_rets = [pid.persistent["ret"] for pid in pids]
            _cat = " ".join(_nohup_rets)
            _command = "while sleep {}; do cat {} && break; done".format(_sleep_duration, _cat)

            self.backend.run(_command)
            return True
        elif all(isinstance(_, (str, int)) for _ in pids):
            pids = [str(_) for _ in pids]
            _flags = "-p ".join([str(pid) for pid in pids])
            _sleep_duration = "{:.1f}".format(refresh_period)
            _command = "while sleep {}; do ps -p {} || break; done".format(
                _sleep_duration, _flags
            )

            self.backend.run(_command)
            return True
        else:
            raise RemoteCommandFailedError(
                "Must supply either pids as integers or strings, or backend persistent commands"
            )

    @property
    def cpuinfo(self) -> dict:
        _sysfs = Path("/sys/devices/system/cpu/cpu[0-9]*")

        def helper(cmd: str, path: t.Union[str, Path], op: t.Callable = lambda x: x):
            _ = self.backend.run(f"{cmd} {_sysfs / path!s}")
            if not _.ok:
                return None
            else:
                return [op(v) for v in _.stdout.split("\n")]

        paths = helper(cmd="ls -d", path="", op=lambda x: Path(x))
        cores = [int(x.stem[3:]) for x in paths]
        online = helper(cmd="cat", path="online", op=lambda x: bool(x))

        if len(online) != len(cores):
            _0 = self.backend.run("cat /sys/devices/system/cpu/online")
            if _0.ok:
                if "0" in _0.stdout:
                    online.insert(0, True)
                else:
                    online.insert(0, False)

        governor = helper(cmd="cat", path="cpufreq/scaling_governor")
        av_freq = helper(
            cmd="cat",
            path="cpufreq/scaling_available_frequencies",
            op=lambda x: sorted([int(_) for _ in x.split()]),
        )
        av_gov = helper(
            cmd="cat", path="cpufreq/scaling_available_governors", op=lambda x: x.split()
        )

        # If any of the above operations fail to produce expected results, provide placeholder
        # values of the appropriate size.
        if governor is None:
            governor = [None] * len(paths)
        if av_freq is None:
            av_freq = [None] * len(paths)
        if av_gov is None:
            av_gov = [None] * len(paths)

        if not all(len(_) == len(paths) for _ in [cores, online, governor, av_gov, av_freq]):
            raise RemoteSetupError("values from sysfs were not of equal length")

        return {
            k: dict(
                path=_p,
                online=_o,
                governor=_g,
                available_governors=_ag,
                available_frequencies=_af,
            )
            for k, _p, _o, _g, _ag, _af in zip(cores, paths, online, governor, av_gov, av_freq)
        }

    @property
    def has_valid_cpufreq(self):
        return getattr(self, "_has_valid_cpufreq", False)

    @has_valid_cpufreq.setter
    def has_valid_cpufreq(self, value):
        if not isinstance(value, bool):
            raise TypeError("Only boolean values can be set")

        self._has_valid_cpufreq = value

    def available_governors(self, cpu: int) -> t.List[str]:
        _cpuinfo = self.cpuinfo
        assert cpu in _cpuinfo, "cpu must be in available cpus"
        return _cpuinfo[cpu]["available_governors"]

    def available_frequencies(self, cpu: int) -> t.List[int]:
        _cpuinfo = self.cpuinfo
        assert cpu in _cpuinfo, "cpu must be in available cpus"
        return _cpuinfo[cpu]["available_frequencies"]

    @property
    def governors(self) -> t.List[str]:
        return [v["governor"] for v in self.cpuinfo.values()]

    @governors.setter
    def governors(self, value: t.Union[t.List[str], str]) -> None:
        _cpuinfo = self.cpuinfo

        if isinstance(value, t.List):
            assert len(value) == len(_cpuinfo.keys()), "governor list and cpu list must match"
            assert all(
                isinstance(_, (type(None), str)) for _ in value
            ), "each value must be str"
        elif isinstance(value, str):
            value = [value] * len(_cpuinfo.keys())

        for i, v in enumerate(value):
            available = _cpuinfo[i]["available_governors"]
            if available:
                if v and v not in [None, *available]:
                    raise DriverValueError(f"{v!r} is not a valid governor ({available!r})")

        for i, v in enumerate(value):
            if v:
                _ = self.backend.run(
                    "echo {!r} | sudo tee {}".format(
                        v, _cpuinfo[i]["path"] / "cpufreq/scaling_governor"
                    )
                )

                if not _.ok:
                    get_root_logger().warning(
                        f"setting governor for cpu {i} failed, exit code {_.exited}, "
                        f"stderr {_.stderr!r}"
                    )

    @property
    def frequencies(self) -> t.List:
        """Read current CPU scaling frequencies for all cores"""
        _cpufreq = Path("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")
        _ = self.backend.run(f"cat {_cpufreq!s}")
        if _.ok:
            return [int(x) for x in _.stdout.split("\n")]
        else:
            return None

    @frequencies.setter
    def frequencies(self, value: t.Optional[t.Union[str, int, t.List]]):
        """Set CPU scaling frequencies for all or specific cores"""
        _cpuinfo = self.cpuinfo

        # if only one value is provided, duplicate the setting across cores
        if not isinstance(value, t.List):
            value = [value] * len(_cpuinfo.keys())
        assert len(value) == len(_cpuinfo.keys()), "frequency list and cpu list must match"

        for i, v in enumerate(value):
            _freq = _cpuinfo[i]["available_frequencies"]
            valid = [None, "", "min", "max", *_freq] if _freq else [None]

            if v:
                if _cpuinfo[i]["governor"] != "userspace":
                    value[i] = None
                    get_root_logger().warning(
                        f"setting frequency for cpu {i} might fail, "
                        f"governor is not 'userspace', but '{_cpuinfo[i]['governor']}',"
                        " value will be set to None"
                    )

            if v == "min":
                value[i] = _cpuinfo[i]["available_frequencies"][0]
            elif v == "max":
                value[i] = _cpuinfo[i]["available_frequencies"][-1]
            elif v == "":
                value[i] = None
            elif v not in valid:
                if isinstance(v, int) and (0.95 * _freq[0] <= v <= 1.05 * _freq[-1]):
                    get_closest = lambda n, c: min(c, key=lambda x: abs(x - n))
                    value[i] = get_closest(v, _freq)
                elif isinstance(v, int) and v < 0.95 & _freq[0]:
                    get_root_logger().warning(
                        f"tried to set frequency {v} which is smaller than the minimum frequency"
                        f"for the platform: {_freq[0]}. Will use the latter."
                    )
                    value[i] = _freq[0]
                elif isinstance(v, int) and v > 1.05 * _freq[-1]:
                    get_root_logger().warning(
                        f"tried to set frequency {v} which is larger than the maximum frequency"
                        f"for the platform: {_freq[1]}. Will use the latter."
                    )
                    value[i] = _freq[1]
                else:
                    raise DriverValueError(f"frequency values ({v}) must be in {valid!r}")

        for i, v in enumerate(value):
            if v:
                _ = self.backend.sudo("cpufreq-set -c {core} -f {freq}".format(core=i, freq=v))
                if not _.ok:
                    get_root_logger().warning(
                        f"setting frequency for cpu {i} failed, exit code {_.exited}, "
                        f"stderr {_.stderr!r}"
                    )

    @property
    def cpuidle(self) -> t.Optional[t.Dict]:
        """Get state-latency pairs for cpuidle

        The values returned by this command
        """
        _idle = Path("/sys/devices/system/cpu/cpu0/cpuidle/state*")

        def helper(cmd: str, path: t.Union[str, Path], op: t.Callable = lambda x: x):
            _ = self.backend.run(f"{cmd} {_idle / path!s}")
            if not _.ok:
                return None
            else:
                return [op(v) for v in _.stdout.split("\n")]

        _states = helper(cmd="cat", path="name")
        _latencies = helper(cmd="cat", path="latency", op=lambda x: int(x))
        return (
            {k: v for k, v in zip(_states, _latencies)}
            if (_states is not None) and (_latencies is not None)
            else None
        )

    @property
    def latency(self) -> t.Optional[int]:
        """Get the CPU DMA latency from devfs

        Returns:
            t.Optional[int]: an integer value or None if unsuccessful
        """
        _cpu_dma_lat = "/dev/cpu_dma_latency"
        # the output is piped to od (octal-dump) with long int decoding,
        # since the raw output is difficult to handle as strings
        _output = self.backend.sudo("cat {} | od -D -An".format(_cpu_dma_lat))

        if _output.ok:
            return int(_output.stdout.strip())
        else:
            return None

    @latency.setter
    def latency(self, value: t.Optional[int]) -> None:
        existing_pgid = getattr(self, "_latency_setter_pgid", None)

        if existing_pgid:
            # must kill whole process group, the pgid of children processes
            # should be the same as calling parent
            if not self.kill(existing_pgid, pgid=True):
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

    @property
    def fan_path(self) -> t.Optional[t.Dict]:
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
        fan = self.fan_path

        if not fan:
            get_root_logger().warning("setting fan failed due to no fan path found")
            return None

        if fan["key"] == "thinkpad":
            _ = self.backend.run(["cat", fan["path"], "|", "grep", "'^level'"])
            return _.stdout.split(":")[1].strip() if _.ok else None

        elif fan["key"] == "odroid-xu3" or fan["key"] == "odroid-2":
            _ = self.backend.run(["cat", fan["path"] + "{fan_mode,pwm_duty}"])
            return tuple(_.stdout.split("\n")) if _.ok else None

        elif fan["key"] == "odroid-xu4":
            _ = self.backend.run(["cat", fan["path"] + "{automatic,pwm1}"])
            return tuple(_.stdout.split("\n")) if _.ok else None

        elif fan["key"] == "pwm-fan":
            _ = self.backend.run(["cat", fan["path"] + "cur_pwm"])
            return _.stdout.strip() if _.ok else None

    @fan.setter
    def fan(self, value: t.Union[str, int, tuple]) -> None:
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

    def getstate(self) -> t.Dict:
        return {
            "governors": self.governors,
            "frequencies": self.frequencies,
            "latency": self.latency,
            "fan": self.fan,
        }

    def setstate(self, **kwargs) -> None:
        if "governors" in kwargs:
            self.governors = kwargs.pop("governors")
        if "frequencies" in kwargs:
            self.frequencies = kwargs.pop("frequencies")
        if "latency" in kwargs:
            self.latency = kwargs.pop("latency")
        if "fan" in kwargs:
            self.fan = kwargs.pop("fan")

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

    def fetch(
        self,
        path_from: t.Union[Path, str],
        path_to: t.Union[Path, str],
        exclude: t.List[str] = Driver.DEFAULT_FETCH_EXCLUDE_LIST,
    ) -> None:
        path_to = _path_type_check(path_to)
        path_from = str(path_from)
        assert check_access(path_to, "w"), "path_to must be writable"
        assert self.exists(path_from), "remote path_from must exist"

        if self.is_dir(path_from):
            path_to = str(path_to) + "/"
            path_from += "/"

        if not isinstance(exclude, t.List):
            raise TypeError("'exclude' must be a list")

        for item in exclude:
            if not isinstance(item, str):
                raise TypeError("'exclude' items must be of type 'str'")

        rsync_args = {
            "key": self.backend.keyfile,
            "port": self.backend.config["port"],
            "user": self.backend.config["user"],
            "host": self.backend.config["ip"],
            "from": path_from,
            "to": path_to,
            "exclude": " ".join([f"--exclude {quote(_)}" for _ in exclude]) if exclude else "",
        }

        if "gateway" in self.backend.config:
            rsync_args.update(gateway=self.backend.config["gateway"])
            command = (
                # "rsync -a -e \"ssh -o 'ProxyCommand nohup ssh -q -A {gateway} nc -q0 %h %p' "
                "rsync -a -e \"ssh -o 'ProxyCommand ssh -q -A {gateway} -W %h:%p' "
                '-i {key} -p {port}" {exclude} {user}@{host}:{from} {to}'
            )
        else:
            command = 'rsync -a -e "ssh -i {key} -p {port}" {exclude} {user}@{host}:{from} {to}'

        command = command.format(**rsync_args)
        return self.backend.connection.local(command, hide=True, warn=True)

    def send(
        self,
        path_from: t.Union[Path, str],
        path_to: t.Union[Path, str],
        exclude: t.List[str] = Driver.DEFAULT_SEND_EXCLUDE_LIST,
    ) -> None:
        path_to = str(path_to)
        path_from = _path_type_check(path_from)
        assert path_from.exists(), "path_from must exist"

        # Copying directories with different combinations of trailing slashes may yield
        # unexpected results. If the `path_from` is a directory, append trailing slashes
        # to both `path_from` and `path_to`.
        if path_from.is_dir():
            path_from = str(path_from) + "/"
            path_to += "/"

        if not self.access(path_to, "w"):
            self.mkdir(str(Path(path_to).parent), parents=True)

        if not isinstance(exclude, t.List):
            raise TypeError("'exclude' must be a list")

        for item in exclude:
            if not isinstance(item, str):
                raise TypeError("'exclude' items must be of type 'str'")

        rsync_args = {
            "key": self.backend.keyfile,
            "port": self.backend.config["port"],
            "user": self.backend.config["user"],
            "host": self.backend.config["ip"],
            "from": path_from,
            "to": path_to,
            "exclude": " ".join([f"--exclude {quote(_)}" for _ in exclude]) if exclude else "",
        }

        if "gateway" in self.backend.config:
            rsync_args.update(gateway=self.backend.config["gateway"])
            command = (
                # "rsync -a -e \"ssh -o 'ProxyCommand nohup ssh -q -A {gateway} nc -q0 %h %p' "
                "rsync -a -e \"ssh -o 'ProxyCommand ssh -q -A {gateway} -W %h:%p' "
                '-i {key} -p {port}" {exclude} {from} {user}@{host}:{to}'
            )
        else:
            command = 'rsync -a -e "ssh -i {key} -p {port}" {exclude} {from} {user}@{host}:{to}'

        command = command.format(**rsync_args)
        return self.backend.connection.local(command, hide=True, warn=True)
