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
"""Concrete adb backends"""

import os
import re
import typing as t
from pathlib import Path
from shlex import quote
from time import sleep

import fabric
import invoke

from exot.exceptions import *
from exot.util.android import Intent
from exot.util.file import check_access, copy, delete
from exot.util.logging import get_root_logger
from exot.util.misc import list2cmdline, validate_helper

from ._backend import Backend

__all__ = ("ADBBackend",)


class ADBBackend(Backend):
    @property
    def required_config_keys(self) -> t.List[str]:
        return ["ip", "user", "port", "key"]

    def validate(self) -> t.NoReturn:
        """Implements `validate` from the Configurable base class"""
        validate_helper(self.config, "ip", str)
        validate_helper(self.config, "user", str)
        validate_helper(self.config, "port", int)
        validate_helper(self.config, "key", str, Path)
        validate_helper(self.config, "gateway", type(None), str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._adb_params = {
            "port": 22350,
            "ip": "127.0.0.1",
            "stdout": "adb-server.out",
            "stderr": "adb-server.err",
            "key": ".adb_key",
        }

        # the local connection
        self._invoke_config = invoke.Config({"run": {"warn": True, "hide": True}})
        self._fabric_config = fabric.Config({"run": {"hide": True, "warn": True}})
        self._connection = invoke.Context(self._invoke_config)
        self._spawned = False
        self._local_adb_version = None
        self._remote_adb_version = None

    @property
    def adb_port(self) -> str:
        return self._adb_params["port"]

    @adb_port.setter
    def adb_port(self, value: str) -> None:
        self._adb_params["port"] = value

    @property
    def adb_ip(self) -> str:
        return self._adb_params["ip"]

    @adb_ip.setter
    def adb_ip(self, value: str) -> None:
        self._adb_params["ip"] = new_ip

    @property
    def adb_stdout(self) -> str:
        return self._adb_params["stdout"] + str(self._adb_params["port"])

    @property
    def adb_stderr(self) -> str:
        return self._adb_params["stdout"] + str(self._adb_params["port"])

    @property
    def adb_key(self) -> str:
        return self._adb_params["key"] + str(self._adb_params["port"])

    @property
    def adb_params(self) -> dict:
        return {
            "port": self.adb_port,
            "ip": self.adb_ip,
            "stdout": self.adb_stdout,
            "stderr": self.adb_stderr,
            "key": self.adb_key,
        }

    @property
    def keyfile(self):
        return getattr(self, "_keyfile", None)

    @keyfile.setter
    def keyfile(self, value):
        _ = Path(os.path.expandvars(value)).expanduser()
        if not _.is_file():
            raise MisconfiguredError("key does not exist", _)
        if not check_access(_):
            raise MisconfiguredError("key not accessible", _)

        self._keyfile = _

    def __getstate__(self):
        # Custom getstate prevents the concrete keyfile being serialised. If key keyfile
        # location resolves to an absolute path, do not serialise it, but rather restore
        # one from the config.
        if self.keyfile and self.keyfile.is_absolute():
            self._keyfile = None

    def __setstate__(self, value):
        if not getattr(self, "_keyfile", None):
            self._keyfile = self.config.key

    @property
    def gateway(self) -> t.Optional[fabric.Connection]:
        return getattr(self, "_gateway", None)

    def _get_adb_version(self, runner: t.Optional[object] = None) -> str:
        runner = self.connection if not runner else runner
        _ = runner.run("adb --version")
        if not _.ok:
            raise BackendRuntimeError(
                f"adb does not seem to be available "
                f"{'locally' if local else 'on gateway'}! {_.stderr}"
            )
        _ = re.match(r".*version ([\d\.]+)$", _.stdout.splitlines()[0])
        if not _:
            raise BackendRuntimeError(
                f"could not determine {'local' if local else 'remote'} adb version!"
            )

        return _.group(1)

    def can_connect(self) -> bool:
        if self.configured:
            # Will throw if the key is not available
            self.keyfile = self.config.key
            self._local_adb_version = self._get_adb_version()

            if "gateway" in self.config:
                try:
                    self._gateway = fabric.Connection(
                        os.path.expandvars(self.config.gateway), config=self._fabric_config
                    )
                    if self.gateway.run("pwd").exited != 0:
                        get_root_logger().critical("Gateway connection failed")
                        return False

                    self._remote_adb_version = self._get_adb_version(self.gateway)

                    if self._local_adb_version != self._remote_adb_version:
                        get_root_logger().critical(
                            (
                                "The ADB version of the gateway server {} does "
                                "not match the client version {}!"
                            ).format(self._remote_adb_version, self._local_adb_version)
                        )
                        return False
                except Exception as e:
                    get_root_logger().critical("Couldn't connect to gateway", e.args)
                    return False

            return True
        else:
            return False

    @property
    def connected(self) -> bool:
        return self._check_connection()

    @property
    def connection(self) -> t.Optional[invoke.Context]:
        return getattr(self, "_connection", None)

    @property
    def is_server_spawned(self):
        return getattr(self, "_spawned", False)

    def _kill_servers(self) -> bool:
        runner = self.gateway if self.gateway else self.connection
        self._spawned
        return runner.run("pkill -u $USER -f '^adb'").ok

    def _spawn_server(self) -> None:
        server_str = "on the gateway" if self.gateway else "locally"
        runner = self.gateway if self.gateway else self.connection

        spawn_new_server = True

        # Copy the adb vendor key over to the gateway or to a local path
        if self.gateway:
            self.gateway.put(self.keyfile, self.adb_key)
        else:
            copy(self.keyfile, self.adb_key, replace=True)

        # Check if ADB server is already running on the port systemwide
        check_server_exists_cmd = runner.run("pgrep -f '^adb'")

        if check_server_exists_cmd.exited == 0:
            pids = check_server_exists_cmd.stdout.rstrip("\r\n").split("\n")
            for server_pid in pids:
                check_server_port_cmd = runner.run(
                    "ps --no-headers -o command {}".format(server_pid)
                )

                regex_match = re.match(r".*(?:-P |tcp:)(\d+)", check_server_port_cmd.stdout)
                if regex_match is None:
                    continue
                _ = regex_match.groups()
                running_server_port = int(_[0]) if _ else -1

                if self.adb_port == running_server_port:
                    if not self._spawned:
                        self.adb_port += 1
                    else:
                        spawn_new_server = False

            if len(pids) > 0:
                if spawn_new_server:
                    get_root_logger().warning(
                        "ADB server already running pids: {pids!r}. Starting server on port {where}".format(
                            pids=pids, where=self.adb_port
                        )
                    )
                else:
                    get_root_logger().warning(
                        "ADB server already running with pids: {pids!r}. Reusing server on port {where}".format(
                            pids=pids, where=self.adb_port
                        )
                    )

        if spawn_new_server:
            cmd = (
                "env ADB_VENDOR_KEYS=$HOME/.adb_key nohup "
                "adb -a -P {port} server nodaemon 1>{stdout} 2>{stderr} & disown"
            ).format(**self.adb_params)

            # Start the ADB server with the specified command
            start_server_cmd = runner.run(cmd)
            if start_server_cmd.exited != 0:
                raise BackendRuntimeError(
                    "Failed to start the ADB server {}".format(server_str),
                    start_server_cmd.stderr,
                )

        if self.gateway:
            self.adb_ip = self.gateway.host

        ps_status = runner.run("pgrep -f -u $USER '^adb.+{port}'".format(**self.adb_params))
        if ps_status.exited != 0:
            raise BackendRuntimeError("ADB server not running {}!".format(server_str))

        self._spawned = True

    def _stop_server(self) -> None:
        # The client can kill the server regardless of its location
        self._adb_wrapper(["kill-server"])
        self._spawned = False

        for adb_file in [self.adb_params[_] for _ in ["key", "stdout", "stderr"]]:
            if self.gateway:
                self.gateway.run("rm -f {}".format(quote(adb_file)))
            else:
                try:
                    delete(Path(os.path.expandvars(adb_file)))
                except:
                    get_root_logger().warning(f"Could not delete {adb_file}")

    @property
    def is_usb(self):
        return self._is_usb(self.config.ip)

    @staticmethod
    def _is_usb(ip):
        return "." not in ip and ip != "localhost"

    def connect(self) -> None:
        if not self.can_connect():
            if not self.config:
                raise CouldNotConnectError("config is missing")
            raise CouldNotConnectError("misconfigured")

        self._spawn_server()
        self._reconnect()

    def _reconnect(self, **kwargs) -> None:
        if self._is_usb(kwargs.get("ip", self.config.ip)):
            self._connect_usb_device(**kwargs)
        else:
            self._connect_tcpip_device(**kwargs)

    def _connect_usb_device(self, **kwargs):
        ip = kwargs.pop("ip", self.config.ip)

        devices = self.devices()
        if ip not in devices:
            raise CouldNotConnectError(
                "Requested serial not available in devices: {!r}".format(devices)
            )
        if self.run(["pwd"], ip=ip).exited != 0:
            raise CouldNotConnectError(
                "Running shell on the target device {} failed!".format(ip)
            )

    def _connect_tcpip_device(self, **kwargs):
        ip = kwargs.pop("ip", self.config.ip)
        port = kwargs.pop("port", self.config.port)

        connect_cmd = self._adb_wrapper(["connect", "{ip}:{port}".format(ip=ip, port=port)])
        if not connect_cmd.ok or "failed" in connect_cmd.stdout:
            raise CouldNotConnectError(
                "Connection via TCP/IP requested but could not connect!",
                connect_cmd.stdout,
                connect_cmd.stderr,
            )

        sleep(2)

        if self.run(["pwd"], ip=ip, port=port).exited != 0:
            raise CouldNotConnectError(
                "Running shell on the target device {} failed!".format(ip)
            )

    def _check_connection(self, **kwargs):
        ip, port = (kwargs.pop("ip", self.config.ip), kwargs.pop("port", self.config.port))

        devices = self.devices()

        if self._is_usb(ip):
            return ip in devices
        else:
            return ip in devices and devices.get(ip, {"port": None})["port"] == port

    def devices(self) -> t.List[str]:
        devices = {}

        devices_cmd = self._adb_wrapper(["devices", "-l"])
        if devices_cmd.exited == 0:
            # Match each alphanumeric serial or ip address and its corresponding description
            for match in re.findall(
                r"^(?!List)([\d\.A-Za-z]+):?(\d*)?\s+(.*)", devices_cmd.stdout, re.MULTILINE
            ):
                devices[match[0]] = dict(
                    [tuple(["port", int(match[1]) if match[1] else -1])]
                    + [tuple(_.split(":")) for _ in match[2].split(" ")][1:]
                )

        return devices

    def disconnect(self) -> None:
        self._adb_wrapper(["disconnect"])
        self._stop_server()

        if self.gateway:
            self.gateway.close()

    def _invoke_wrapper(self, cmd: Backend.CommandT, **kwargs: t.Any) -> Backend.ReturnT:
        if not isinstance(cmd, (str, t.List)):
            raise TypeError(f"'cmd' ({type(cmd)}) not str or list")
        if isinstance(cmd, t.List):
            cmd = list2cmdline(cmd)

        # hide: do not output to console
        if "hide" not in kwargs:
            kwargs["hide"] = True
        # warn: do not throw when return code != 0
        if "warn" not in kwargs:
            kwargs["warn"] = True

        if "history" in kwargs:
            history = kwargs.pop("history")
            if not isinstance(history, bool):
                _ = f"keyword argument 'history' should be bool, got: {type(history)}"
                raise TypeError(_)
        else:
            history = True

        _ = self.connection.run(cmd, **kwargs)
        _.stdout = _.stdout.rstrip("\r\n")
        _.stderr = _.stderr.rstrip("\r\n")

        _.stdout = _.stdout.replace("\r", "")
        _.stderr = _.stderr.replace("\r", "")

        if history:
            self._history.append(_)

        return _

    def _adb_wrapper(self, cmd: Backend.CommandT, **kwargs):
        if not isinstance(cmd, (str, t.List)):
            raise TypeError(f"'cmd' ({type(cmd)}) not str or list")

        # -L points the adb client to the adb server
        pre = ["adb", "-L", "tcp:{ip}:{port}".format(**self.adb_params)]
        post = kwargs.pop("post", [])

        if not isinstance(post, list):
            raise TypeError("'post' can only handle commands in list form")

        if isinstance(cmd, t.List):
            cmd = pre + cmd + post
        else:
            cmd = " ".join([list2cmdline(pre), cmd, list2cmdline(post)])

        return self._invoke_wrapper(cmd, **kwargs)

    def _adb_device_wrapper(self, cmd: Backend.CommandT, **kwargs):
        if not isinstance(cmd, (str, t.List)):
            raise TypeError(f"'cmd' ({type(cmd)}) not str or list")

        ip, port = (kwargs.pop("ip", self.config.ip), kwargs.pop("port", self.config.port))
        is_usb = self._is_usb(ip)

        pre = ["-s", f"{ip}:{port}" if not is_usb else ip]

        if isinstance(cmd, t.List):
            cmd = pre + cmd
        else:
            cmd = " ".join([list2cmdline(pre), cmd])

        if not is_usb:

            def connection_is_ok():
                return self._adb_wrapper(pre + ["shell", "pwd"], history=False).ok

            if not connection_is_ok():
                get_root_logger().critical(
                    f"TCP/IP ADB connection to {ip}:{port} went offline, reconnecting..."
                )

                # Make sure the adb server is running
                self._spawn_server()

                # Try to reconnect using exponentiall back-off waiting time
                max_retries = 5
                for retries in range(1, max_retries):
                    sleep(2 ** (retries) - 1)
                    self._adb_wrapper(["connect", f"{ip}:{port}"], history=False)
                    if connection_is_ok():
                        break
                    else:
                        # Try to restart the adb connection. Sometimes even if the connection seems not ok
                        # we can still access the device as adb is simply messed up but not fully broken.
                        self._adb_wrapper(
                            ["shell", "su", "-c", "setprop", "ctl.restart", "adbd"]
                        )
                        if retries == (max_retries - 1):
                            # As a last resort, try to reboot the phone
                            sleep(2)
                            self._adb_wrapper(["shell", "su", "-c", "reboot"])
                            sleep(60 * 5)  # Wait for five minutes until the phone is back up

                if not connection_is_ok():
                    raise BackendRuntimeError(
                        f"Could not re-establish TCP/IP ADB connection to {ip}:{port} after "
                        f"{retries} retries."
                    )

        return self._adb_wrapper(cmd, **kwargs)

    def run(self, cmd: Backend.CommandT, **kwargs) -> Backend.ReturnT:
        """
        As ADB often does not properly forward the return values of commands, we
        append the return value to the stdout stream. This function also parses
        the return value from stdout to ensure compatibiliy. We do this for all commands
        except the activity manager (am), as quoting creates issues with the extras.
        """
        if not isinstance(cmd, (str, t.List)):
            raise TypeError(f"'cmd' ({type(cmd)}) not str or list")

        if "am" in cmd or "input" in cmd:
            pre = ["shell"]
            if isinstance(cmd, t.List):
                cmd = [*pre, *cmd]
            else:
                cmd = " ".join([list2cmdline(pre), quote(cmd)])
            return_cmd = self._adb_device_wrapper(cmd, **kwargs)
        else:
            retval = [";", "echo", "$?"]
            if isinstance(cmd, t.List):
                cmd = [*cmd, *retval]
            else:
                cmd = cmd.split(" ")
                cmd = [*cmd, *retval]

            pre = ["shell"]
            if isinstance(cmd, t.List):
                cmd = " ".join([list2cmdline(pre), quote(list2cmdline(cmd))])
            else:
                cmd = " ".join([list2cmdline(pre), quote(cmd)])

            return_cmd = self._adb_device_wrapper(cmd, **kwargs)
            if len(return_cmd.stdout) > 0:
                if return_cmd.stdout[-1].isdigit():
                    return_cmd.exited = int(return_cmd.stdout[-1])
                    return_cmd.stdout = return_cmd.stdout.rstrip(str(return_cmd.exited))
                    return_cmd.stdout = return_cmd.stdout.rstrip("\n")
                    return_cmd.stdout = return_cmd.stdout.rstrip("\r")

        return return_cmd

    def sudo(self, cmd: Backend.CommandT, **kwargs) -> Backend.ReturnT:
        if not isinstance(cmd, (str, t.List)):
            raise TypeError(f"'cmd' ({type(cmd)}) not str or list")

        if isinstance(cmd, t.List):
            su_cmd = ["su", "-c", list2cmdline(cmd)]
        else:
            su_cmd = " ".join(["su -c", quote((cmd))])

        return self.run(su_cmd, **kwargs)

    def send_intent(self, intent: Intent, kind: str = "service") -> Backend.ReturnT:
        if not isinstance(intent, Intent):
            raise TypeError("Requires an Intent instance")
        if kind not in ["service", "activity", "broadcast"]:
            raise ValueError("'kind' must be either service, activity, or broadcast")

        if not intent:
            raise ValueError("Empty intent")

        kind_to_command = {
            "service": "startservice",
            "activity": "start",
            "broadcast": "broadcast",
        }

        intent_cmd_result = self.run(["am", kind_to_command[kind], *intent.assemble()])

        if intent_cmd_result.stderr:
            intent_cmd_result.exited = -1

        return intent_cmd_result
