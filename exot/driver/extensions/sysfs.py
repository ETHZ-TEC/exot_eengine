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
import abc
import typing as t
from pathlib import Path

from exot.exceptions import *
from exot.util.logging import get_root_logger


class StateMethods(metaclass=abc.ABCMeta):
    @property
    def cpuinfo(self) -> dict:
        paths, cores = self._get_cpu_sysfs_paths_and_cores()
        online = self.online_cores
        governor = self.governors
        av_freq = self.available_frequencies
        av_gov = self.available_governors

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
    def available_governors(self) -> t.List[str]:
        return self._cpu_sysfs_seq_helper(
            cmd="cat", path="cpufreq/scaling_available_governors", op=lambda x: x.split()
        )

    @property
    def available_frequencies(self) -> t.List[int]:
        return self._cpu_sysfs_seq_helper(
            cmd="cat",
            path="cpufreq/scaling_available_frequencies",
            op=lambda x: sorted([int(_) for _ in x.split()]),
        )

    def _get_cpu_sysfs_paths_and_cores(self) -> t.List[Path]:
        cmd_result = self.command(f"ls -d /sys/devices/system/cpu/cpu[0-9]*")
        paths = [Path(_) for _ in cmd_result.stdout.splitlines()]
        cores = [int(x.stem[3:]) for x in paths]

        return paths, cores

    def _cpu_sysfs_glob_helper(
        self, cmd: str, path: t.Union[str, Path], op: t.Callable = lambda x: x
    ):
        _sysfs = Path("/sys/devices/system/cpu/cpu[0-9]*")
        _ = self.command(f"{cmd} {_sysfs / path!s}")
        return [op(v) for v in _.stdout.splitlines() if v] if _.ok else None

    def _cpu_sysfs_idx_helper(
        self, cmd: str, cpu: int, path: t.Union[str, Path], op: t.Callable = lambda x: x
    ):
        _path = Path("/sys/devices/system/cpu/cpu{}".format(cpu)) / path
        _ = self.command(f"{cmd} {_path!s}")
        return op(_.stdout) if _.ok else None

    def _cpu_sysfs_seq_helper(
        self, cmd: str, path: t.Union[str, Path], op: t.Callable = lambda x: x
    ):
        _, cores = self._get_cpu_sysfs_paths_and_cores()
        value = [None] * len(cores)

        for core in cores:
            value[core] = self._cpu_sysfs_idx_helper(cmd, core, path, op)

        return value

    @property
    def online_cores(self) -> t.List[bool]:
        online = self._cpu_sysfs_seq_helper(
            cmd="cat", path="online", op=lambda x: x.strip() == "1"
        )

        if online[0] is None:
            _0 = self.command("cat /sys/devices/system/cpu/online")
            if _0.ok:
                if "0" in _0.stdout:
                    online.insert(0, True)
                else:
                    online.insert(0, False)

        return online

    @online_cores.setter
    def online_cores(self, value: t.List[t.Union[bool, None]]) -> None:
        paths, _ = self._get_cpu_sysfs_paths_and_cores()

        if len(value) != len(paths):
            raise DriverValueError(
                "online cores list does not match the number of available cores"
            )

        for idx, v in enumerate(value):
            if v is not None:
                self.command(
                    [
                        "su",
                        "-c",
                        list2cmdline(
                            ["echo", "1" if v else "0", "|", "tee", str(paths[idx] / "online")]
                        ),
                    ],
                    sudo=True,
                )

    @property
    def governors(self) -> t.List[str]:
        return self._cpu_sysfs_seq_helper(cmd="cat", path="cpufreq/scaling_governor")

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
                _ = self.command(
                        "sudo su -c 'echo {!r} | tee {}'".format(
                        v, _cpuinfo[i]["path"] / "cpufreq/scaling_governor"
                    ),
                    sudo=True,
                )

                if not _.ok:
                    get_root_logger().warning(
                        f"setting governor for cpu {i} failed, exit code {_.exited}, "
                        f"stderr {_.stderr!r}"
                    )

    @property
    def frequencies(self) -> t.List:
        """Read current CPU scaling frequencies for all cores"""

        def _freq_helper(value):
            try:
                return int(value.strip())
            except (ValueError, TypeError):
                return None

        return self._cpu_sysfs_seq_helper(
            cmd="cat", path="cpufreq/scaling_cur_freq", op=_freq_helper
        )

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
                elif isinstance(v, int) and v < 0.95 * _freq[0]:
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

        has_cpufreq_set = self.command("command -v cpufreq-set").ok
        for i, v in enumerate(value):
            if v:
                _ = (
                    self.command(
                        "cpufreq-set -c {core} -f {freq}".format(core=i, freq=v), sudo=True
                    )
                    if has_cpufreq_set
                    else self.command(
                        (
                            "su -c 'echo {freq} | tee /sys/devices/system/cpu/cpu{core}/"
                            "cpufreq/scaling_setspeed'"
                        ).format(core=i, freq=v),
                        sudo=True,
                    )
                )
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
            _ = self.command(f"{cmd} {_idle / path!s}")
            if not _.ok:
                return None
            else:
                return [op(v) for v in _.stdout.splitlines() if v]

        _states = helper(cmd="cat", path="name")
        _latencies = helper(cmd="cat", path="latency", op=lambda x: int(x))
        return (
            {k: v for k, v in zip(_states, _latencies)}
            if (_states is not None) and (_latencies is not None)
            else None
        )
