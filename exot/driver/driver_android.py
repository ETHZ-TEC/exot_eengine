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
"""Concrete Android drivers"""
from __future__ import annotations

import enum
import json
import re
import tempfile
import typing as t
from datetime import datetime, timedelta
from pathlib import Path
from shlex import quote
from time import sleep

from exot.exceptions import *
from exot.util.android import Intent
from exot.util.file import _path_type_check, check_access, delete
from exot.util.logging import get_root_logger

from . import extensions
from ._driver import Driver
from ._process import Process
from .backend_adb import ADBBackend
from .extensions.android import *

__all__ = ("ADBAndroidDriver", "EXOT_APP_ACTIONS", "EXOT_APP_KEYS", "EXOT_APP_STATE")

EXOT_APP_ACTIONS = {
    "start": EXOT_NAMESPACE + ".intents.service.action.START",
    "stop": EXOT_NAMESPACE + ".intents.service.action.STOP",
    "create": EXOT_NAMESPACE + ".intents.service.action.CREATE",
    "reset": EXOT_NAMESPACE + ".intents.service.action.RESET",
    "destroy": EXOT_NAMESPACE + ".intents.service.action.DESTROY",
    "query": EXOT_NAMESPACE + ".intents.service.action.QUERY",
}

EXOT_APP_KEYS = {
    "config": EXOT_NAMESPACE + ".intents.service.key.CONFIG",
    "status": EXOT_NAMESPACE + ".intents.service.key.STATUS",
}


@enum.unique
class EXOT_APP_STATE(enum.Enum):
    MISSING = "missing"
    IDLE = "idle"
    STARTED = "started"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    INVALID = "invalid"


@enum.unique
class STANDALONE_APP_STATE(enum.Enum):
    UNKNOWN = "unknown"


class ADBAndroidDriver(
    extensions.android.AndroidMethods,
    extensions.sysfs.StateMethods,
    extensions.unix.LatencyMethods,
    extensions.unix.LockingMethods,
    extensions.unix.ExtraFileMethods,
    extensions.unix.FileMethods,
    Driver,
    backend=ADBBackend,
):
    unix_signal = extensions.unix.ProcessMethods.unix_signal
    unix_wait = extensions.unix.ProcessMethods.unix_wait

    @property
    def lock_file(self):
        return "/sdcard/exot.lock"

    @property
    def getable_states(self):
        return [
            "governors",
            "frequencies",
            "latency",
            "power_state",
            "focused_window",
            "focused_activity",
            "installed_packages",
            "running_activities",
            "running_services",
            "recent_activities",
            "active_windows",
            "is_screen_locked",
            "is_screen_on",
            "is_home_screen",
        ]

    @property
    def setable_states(self):
        return ["governors", "frequencies"]

    @property
    def latency(self):
        return super().latency

    @latency.setter
    def latency(self, value: t.Any):
        """
        No way of running a persistent background process so far, setter is disabled
        """
        pass

    def executable_exists(self, executable: str) -> bool:
        if not isinstance(executable, str):
            raise TypeError("executable", type(executable), str)
        if executable.split("/")[0] in self.list_packages():
            return True
        else:
            return self.backend.run(["which", executable]).ok

    def _spawn_framework_app(
        self, path: str, args: t.List[str], details: t.Dict, slaves: t.List[Process] = []
    ) -> t.Tuple[Process, ADBBackend.ReturnT]:
        if not isinstance(args, t.List):
            raise TypeError("'args' should be a list")
        if not self.executable_exists(path):
            raise RemoteCommandFailedError(f"{path!r} not executable")

        if not hasattr(details, "json") or len(args) < 2:
            raise RuntimeError("Framework apps require the JSON to be available")

        if hasattr(details, "json"):
            config = details.json
        else:
            try:
                config = args[2]
                json.loads(config)
            except json.JSONDecodeError:
                raise RuntimeError("Malformatted JSON configuration!")

        intent = Intent(
            COMPONENT_NAME=path,
            ACTION=EXOT_APP_ACTIONS["create"],
            es={EXOT_APP_KEYS["config"]: config},
        )

        # Send the starting intent
        intent_cmd = self.send_intent(intent)
        sleep(2)
        # Create the invocation wrapper
        invocation = Invocation(
            driver=self,
            is_framework_app=True,
            component=path,
            details=details,
            spawned_at=datetime.now(),
            duration=details.duration,
        )
        # Create the process wrapper
        process = Process(
            driver=self,
            invocation=invocation,
            identity=path,
            slaves=slaves,
            duration=details.duration,
        )

        return process, intent_cmd

    def _spawn_generic_app(
        self, path: str, details: t.Dict, slaves: t.List[Process] = []
    ) -> t.Tuple[Process, ADBBackend.ReturnT]:
        component_from_intent = (
            details.intent.COMPONENT_NAME if details.intent is not None else None
        )
        component = path if "/" in path else component_from_intent

        # For generic apps, we only send an startService intent with an empty action and
        # empty extras to have the service started.
        intent = Intent(COMPONENT_NAME=component)
        intent_cmd = self.send_intent(intent)

        # Create the invocation wrapper
        invocation = Invocation(
            driver=self,
            is_framework_app=False,
            component=component,
            details=details,
            spawned_at=datetime.now(),
            duration=details.duration,
        )
        # Create the process wrapper
        process = Process(
            driver=self,
            invocation=invocation,
            identity=component,
            slaves=slaves,
            duration=details.duration,
        )

        return process, intent_cmd

    def spawn(
        self,
        path: str,
        args: t.List[str],
        slaves: t.List[Process] = [],
        details: t.Optional[t.Dict] = None,
    ) -> t.Tuple[Process, ADBBackend.ReturnT]:
        if not details:
            raise RuntimeError(
                "Android spawning requires detailed app information passed in 'details'"
            )

        if details.standalone:
            process, ret = self._spawn_generic_app(path, details, slaves)
        else:
            process, ret = self._spawn_framework_app(path, args, details, slaves)

        return process, ret

    def _start_framework_app(self, process):
        intent = Intent(COMPONENT_NAME=process.identity, ACTION=EXOT_APP_ACTIONS["start"])
        process.invocation.started_at = datetime.now()
        return self.send_intent(intent, "service").ok

    def _start_generic_app(self, process):
        process.invocation.started_at = datetime.now()
        return self.send_intent(process.invocation.details.intent, "service").ok

    def start(self, *processes: Process, **kwargs) -> bool:
        status = True
        for process in processes:
            status &= (
                self._start_framework_app(process)
                if process.invocation.is_framework_app
                else self._start_generic_app(process)
            )

        return status

    def _stop_framework_app(self, process):
        intent = Intent(COMPONENT_NAME=process.identity, ACTION=EXOT_APP_ACTIONS["stop"])
        return self.send_intent(intent, "service").ok

    def _stop_generic_app(self, process):
        return self.force_stop_package(process.identity.split("/")[0])

    def stop(self, *processes: Process, **kwargs) -> bool:
        status = True
        for process in processes:
            status &= (
                self._stop_framework_app(process)
                if process.invocation.is_framework_app
                else self._stop_generic_app(process)
            )

    def _kill_framework_app(self, process):
        intent = Intent(COMPONENT_NAME=process.identity, ACTION=EXOT_APP_ACTIONS["destroy"])
        _ = self.send_intent(intent, "service").ok
        return _ & self.force_stop_package(process.identity.split("/")[0])

    def _kill_generic_app(self, process):
        return self.force_stop_package(process.identity.split("/")[0])

    def kill(self, *processes: Process, **kwargs) -> bool:
        status = True
        for process in processes:
            status &= (
                self._kill_framework_app(process)
                if process.invocation.is_framework_app
                else self._kill_generic_app(process)
            )

    def wait(self, *processes: Process, **kwargs) -> bool:
        for process in processes:
            sleep_time = max(kwargs.get("refresh_period", 1.0), 1.0)

            while True:
                if process.exited is not None:
                    break
                sleep(sleep_time)

        return True

    @property
    def reboot_count(self) -> int:
        if not hasattr(self, "_reboot_count"):
            self._reboot_count = 0
        return self._reboot_count

    @reboot_count.setter
    def reboot_count(self, value: int):
        if isinstance(value, int):
            self._reboot_count = value
        else:
            raise TypeError(f"reboot_count has to be of type int, but is {type(value)}")

    def reboot_device(self):
        get_root_logger().info("Rebooting device, this will take roughly 2 minutes...")
        self.backend.sudo("reboot")
        sleep(90)  # 1.5 minutes security wait time until the device is back online

    def initstate(self) -> None:
        # Check reboot count, if it is bigger than or equal 5, reboot the device
        # This helps to prevent adb/android congestion. Tests have shown that after
        # some online time, the android device does not respond properly anymore.
        if self.reboot_count >= 5:
            self.reboot_count = 0
            self.reboot_device()
        self.reboot_count = self.reboot_count + 1
        self.turn_screen_on()
        sleep(0.5)
        self.unlock_screen()
        self.open_home_screen()
        sleep(0.5)
        self.clear_app_switcher()
        sleep(0.5)
        self.clear_chrome_tabs()
        self.backend.sudo("pm clear com.google.android.youtube")

        if not self.is_proxy_service_active:
            self.start_proxy_service()
        self.open_home_screen()

    def cleanup(self):
        if self.original_state:
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
            self.setstate(**_)  # restore state
        self.initstate()
        self.reboot_device()
        self.turn_screen_off()

    def fetch(
        self,
        path_from: t.Union[Path, str],
        path_to: t.Union[Path, str],
        exclude: t.List[str] = Driver.DEFAULT_FETCH_EXCLUDE_LIST,
    ) -> ADBBackend.ReturnT:
        """Fetches data from the device

        Due to how ADB file transfer with 'push' and 'pull' works, if the source path is a
        directory, we first make a temporary local directory (with `tempfile.mkdtemp`), then
        transfer the entire directory from the source there, and then use rsync locally
        to allow filtering out and synchronising directories.

        Args:
            path_from (t.Union[Path, str]): The path to a file/folder on the device
            path_to (t.Union[Path, str]): The path to the destination file/folder
            exclude (t.List[str], optional): The exclusion list
        """
        path_to = _path_type_check(path_to)
        path_from = str(path_from)

        if not check_access(path_to, "w"):
            raise TransferFailedError("path_to must be writable")
        if not self.exists(path_from):
            raise TransferFailedError("remote path_from must exist")

        if not self.is_dir(path_from):
            return self.backend._adb_device_wrapper(["pull", path_from, str(path_to)])
        else:
            path_to = str(path_to) + "/"
            path_from += "/"

            intermediate_path = Path(tempfile.mkdtemp(prefix="exot-fetch-"))
            self.backend._adb_device_wrapper(["pull", path_from, str(intermediate_path)])

            rsync_args = {
                "from": str(intermediate_path) + "/" + path_from.split("/")[-2] + "/",
                "to": path_to,
                "exclude": " ".join([f"--exclude {quote(_)}" for _ in exclude])
                if exclude
                else "",
            }

            rsync_cmd_result = self.backend.connection.run(
                "rsync -a {exclude} {from} {to}".format(**rsync_args)
            )
            delete(intermediate_path)

            return rsync_cmd_result

    def send(
        self,
        path_from: t.Union[Path, str],
        path_to: t.Union[Path, str],
        exclude: t.List[str] = Driver.DEFAULT_SEND_EXCLUDE_LIST,
    ) -> ADBBackend.ReturnT:
        """Sends data to the device

        Due to how ADB file transfer with 'push' and 'pull' works, if the local path is a
        directory, we first make a temporary local directory (with `tempfile.mkdtemp`), then
        use rsync locally to apply exclusion list and  synchronise with the source directory,
        and then transfer the temporary directory to the device.

        Args:
            path_from (t.Union[Path, str]): The local file/folder
            path_to (t.Union[Path, str]): The path to the destination file/folder
            exclude (t.List[str], optional): The exclusion list
        """
        path_to = str(path_to)
        path_from = _path_type_check(path_from)
        path_from = path_from.expanduser()
        if not path_from.exists():
            raise RuntimeError(f"path {path_from!s} does not exist")

        # Pushing works strangely with adb push, not like rsync
        if path_from.is_file():
            return self.backend._adb_device_wrapper(["push", str(path_from), path_to])
        # Build folder to send by executing rsync with exclusion list
        else:
            intermediate_path = Path(tempfile.mkdtemp(prefix="exot-send-"))

            rsync_args = {
                "from": path_from,
                "to": intermediate_path,
                "exclude": " ".join([f"--exclude {quote(_)}" for _ in exclude])
                if exclude
                else "",
            }

            rsync_result = self.backend.connection.run(
                "rsync -a {exclude} {from}/ {to}/".format(**rsync_args)
            )
            invoke_command_result = self.backend._adb_device_wrapper(
                ["push", str(intermediate_path), "/sdcard/"]
            )
            delete(intermediate_path)

            intermediate_remote_path = str(Path("/sdcard/") / intermediate_path.name)
            if not self.exists(intermediate_remote_path):
                raise TransferFailedError(
                    f"temporary remote directory {intermediate_remote_path} "
                    "does not exist after pushing!"
                )

            self.mkdir(path_to)
            self.copy(intermediate_remote_path + "/*", path_to, recursive=True)

            self.delete(intermediate_remote_path, recursive=True)

            return invoke_command_result


class Invocation:
    def __init__(
        self,
        driver: ADBAndroidDriver,
        component: str,
        is_framework_app: bool,
        details: t.Dict,
        spawned_at: datetime = datetime.now(),
        duration: t.Optional[float] = None,
        exit_on_expiration: bool = True,
    ):
        """Creates the invocation object

        Args:
            driver (ADBAndroidDriver): The driver
            component (str): The component/identity
            is_framework_app (bool): Is this a framework app?
            spawned_at (datetime, optional): Spawning time point
            duration (t.Optional[float], optional): Estimated duration?
            exit_on_expiration (bool, optional): Should report as exited once estimated duration is reached?
        """
        self.driver = driver
        self.component = component
        self.is_framework_app = is_framework_app
        self.details = details
        self.spawned_at = spawned_at
        self.duration = timedelta(seconds=duration) if duration else None
        self._exit_on_expiration = exit_on_expiration
        self._expiration_time = None
        self._current_state = self.query_state()
        self._search_str = (
            r".*(exot|ExOT|log\s+:|app\s+:).*"
            if self.is_framework_app
            else r".*" + self.component.split("/")[0] + r".*"
        )

    def __repr__(self) -> str:
        return "<{} component={} spawned_at={} running_time={} state={} at {}>".format(
            self.__class__.__name__,
            self.component,
            self.spawned_at.isoformat(timespec="seconds"),
            self.running_time,
            self._current_state,
            hex(id(self)),
        )

    @property
    def active_time(self) -> timedelta:
        """Gets the active time (since spawning)

        Returns:
            timedelta: The active time
        """
        return datetime.now() - self.spawned_at

    @property
    def running_time(self) -> t.Optional[timedelta]:
        """Gets the running time (since started)

        Returns:
            t.Optional[timedelta]: The running time
        """
        return datetime.now() - self.started_at if self.started_at else None

    @property
    def remaining_time(self) -> t.Optional[timedelta]:
        """Gets the remaining time

        Returns:
            t.Optional[timedelta]: Remaining time if expiration time is set
        """
        if self._expiration_time is not None:
            return self._expiration_time - datetime.now()
        else:
            return None

    @property
    def started_at(self) -> datetime:
        return getattr(self, "_started_at", None)

    @started_at.setter
    def started_at(self, value) -> None:
        self._started_at = value
        self._expiration_time = self._started_at + self.duration if self.duration else None

    @property
    def ok(self) -> bool:
        """Exited correctly?

        Returns:
            bool: True if exited with zero exit code
        """
        if self.exited is None:
            return False
        else:
            return self.exited <= 0

    @property
    def exited(self) -> t.Optional[int]:
        """Gets the exit code of the process

        Returns:
            t.Optional[int]: The exit code or None if not exited
        """
        if self._exit_on_expiration and self._expiration_time is not None:
            return -1 if self.remaining_time <= timedelta(0) else None

        if self.is_framework_app:
            return (
                -2
                if self._current_state
                in [EXOT_APP_STATE.INVALID, EXOT_APP_STATE.MISSING, EXOT_APP_STATE.TERMINATED]
                else None
            )
        else:
            services = self.driver.list_services(with_filter=lambda record: self.component)
            return -3 if not services else None

    @property
    def stderr(self) -> str:
        """Gets the "standard error"

        Returns:
            str: The "standard error" output, i.e. the log lines that match the self._search_str
        """
        return "\n".join(
            filter(
                lambda x: re.match(self._search_str, x) is not None,
                self.driver.get_logs(since=self.spawned_at),
            )
        )

    @property
    def stdout(self) -> str:
        return ""

    def query_state(self) -> t.Union[EXOT_APP_STATE, STANDALONE_APP_STATE]:
        """Queries the state

        Returns:
            t.Union[EXOT_APP_STATE, STANDALONE_APP_STATE]: The state
        """
        if self.is_framework_app:
            return self._query_state_framework()
        else:
            return STANDALONE_APP_STATE.UNKNOWN

    def _query_state_framework(self) -> EXOT_APP_STATE:
        """Queries the state of a framework app process

        Returns:
            EXOT_APP_STATE: The process state
        """

        query_intent = Intent(COMPONENT_NAME=self.component, ACTION=EXOT_APP_ACTIONS["query"])
        intent_time = self.driver.get_time()
        self.driver.send_intent(query_intent)

        logs = self.driver.get_logs(grep=r"handleActionQuery\(\): (\S+)", since=intent_time)

        if len(set(logs)) > 1:
            get_root_logger().warning(
                f"logs returned more than one state query output: {set(logs)}"
            )

        return EXOT_APP_STATE(logs[0]) if logs else EXOT_APP_STATE.INVALID

    def update(self):
        """Updates the state
        """
        self._current_state = self.query_state()
