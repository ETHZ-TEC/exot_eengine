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
import re
import typing as t
from copy import copy
from datetime import datetime
from time import sleep

import numpy as np

from exot.util.android import *

from .._backend import CommandT, ReturnT

__all__ = ("AndroidMethods", "EXOT_NAMESPACE", "INTENTPROXY_COMPONENT", "INTENTPROXY_ACTIONS")


EXOT_NAMESPACE = ".".join(["ch", "ethz", "exot"])
INTENTPROXY_NAMESPACE = EXOT_NAMESPACE  # ".".join(["ch", "ethz", "karajan"])
INTENTPROXY_COMPONENT = INTENTPROXY_NAMESPACE + ".intentproxy/.IntentProxyService"

action_prefix = INTENTPROXY_NAMESPACE + ".intents.IntentProxy.action."
keyextra_prefix = INTENTPROXY_NAMESPACE + ".intents.IntentProxy.keyextra."

INTENTPROXY_ACTIONS = {
    "bundle_extras": action_prefix + "BUNDLE_EXTRAS",
    "start_apps": action_prefix + "START_APPS",
    "stop_apps": action_prefix + "STOP_APPS",
    "configure_app": action_prefix + "CONFIGURE_APP",
    "forward_": action_prefix + "FORWARD_",
    "forward": action_prefix + "FORWARD",
    "forward_bundle": action_prefix + "FORWARD_BUNDLE",
    "forward_startservice": action_prefix + "FORWARD_STARTSERVICE",
    "forward_startactivity": action_prefix + "FORWARD_STARTACTIVITY",
}
INTENTPROXY_KEYS = {
    "intent_action": keyextra_prefix + "intent.action",
    "intent_component": keyextra_prefix + "intent.component",
    "intent_flags": keyextra_prefix + "intent.flags",
    "intent_extras_key": keyextra_prefix + "intent.extra.key",
    "default_bundle": keyextra_prefix + "BUNDLE",
    "app_array": keyextra_prefix + "APPS_ARRAY",
}


class AndroidMethods(metaclass=abc.ABCMeta):
    def list_packages(self, with_filter: t.Optional[t.Callable] = None) -> t.List[str]:
        """Lists packages available on the platform

        Args:
            with_filter (t.Optional[t.Callable], optional): Filtering callable, (str) -> bool

        Returns:
            t.List[str]: A list of packages
        """
        value = [
            _.split(":")[-1] for _ in self.backend.run("pm list packages").stdout.splitlines()
        ]

        return value if with_filter is None else list(filter(with_filter, value))

    def list_dumpsys_services(self) -> t.List[str]:
        """Lists services available via platform's dumpsys

        Returns:
            t.List[str]: A list of services/commands
        """
        return [
            _.strip()
            for _ in self.backend.run("dumpsys -l").stdout.rstrip("\r\n").splitlines()[1:]
        ]

    def _list_record_helper(
        self,
        command: CommandT,
        record: t.Type[t.T] = None,
        with_filter: t.Optional[t.Callable] = None,
        with_stack: bool = False,
        **kwargs: t.Any,
    ) -> t.List[t.T]:
        """Lists records of a particular type for a given Command

        Args:
            command (CommandT): The command to run on the target Android device
            record (t.Type[t.T], optional): The record type, e.g. TaskRecord or ActivityRecord
            with_filter (t.Optional[t.Callable], optional): Filtering callable, (Record) -> bool
            with_stack (bool, optional): Split and parse per stack id?
            **kwargs (t.Any): Other keyword arguments passed to `RecordParser.parse_record`

        Returns:
            t.List[t.T]: A list of records
        """
        cmd_result = self.backend.run(command).stdout.rstrip("\r\n")
        accumulator = []

        if with_stack:
            for stack_block in re.split(r"Stack #|mFocusedActivity", cmd_result)[1:-1]:
                regex_match = re.match(r"^\d+", stack_block.replace("\r", ""))
                if regex_match is None:
                    continue
                else:
                    stack = re.match(r"^\d+", stack_block.replace("\r", "")).group()
                kwargs["stackId"] = stack
                accumulator += [
                    RecordParser.parse_record(
                        line, record, exact=kwargs.pop("exact", False), **kwargs
                    )
                    if record is not None
                    else RecordParser.parse_any_record(
                        line, exact=kwargs.pop("exact", False), **kwargs
                    )
                    for line in stack_block.splitlines()
                ]
        else:
            accumulator = [
                RecordParser.parse_record(
                    line, record, exact=kwargs.pop("exact", False), **kwargs
                )
                if record is not None
                else RecordParser.parse_any_record(
                    line, exact=kwargs.pop("exact", False), **kwargs
                )
                for line in cmd_result.splitlines()
            ]

        # Activities are often reported from top to bottom, it's desirable to preserve the order.
        value = [_ for _ in sorted(set(accumulator), key=accumulator.index) if _ is not None]

        if value is None:
            value = [WindowRecord("", "", "")]

        return value if with_filter is None else list(filter(with_filter, value))

    def list_tasks(self, **kwargs) -> t.List[TaskRecord]:
        cmd = "dumpsys activity activities"
        return self._list_record_helper(cmd, TaskRecord, with_stack=True, **kwargs)

    def list_activities(self, **kwargs) -> t.List[ActivityRecord]:
        cmd = "dumpsys activity activities"
        return self._list_record_helper(cmd, ActivityRecord, with_stack=True, **kwargs)

    def list_services(self, **kwargs) -> t.List[ServiceRecord]:
        cmd = "dumpsys activity services"
        return self._list_record_helper(cmd, ServiceRecord, **kwargs)

    def list_recent_tasks(self, **kwargs) -> t.List[TaskRecord]:
        cmd = "dumpsys activity recents"
        return self._list_record_helper(cmd, TaskRecord, **kwargs)

    def list_recent_activities(self, **kwargs) -> t.List[ActivityRecord]:
        cmd = "dumpsys activity recents"
        return self._list_record_helper(cmd, ActivityRecord, **kwargs)

    def list_windows(self, **kwargs) -> t.List[WindowRecord]:
        cmd = "dumpsys window windows"
        return self._list_record_helper(cmd, WindowRecord, **kwargs)

    @property
    def installed_packages(self):
        return self.list_packages()

    @property
    def running_activities(self):
        return self.list_activities()

    @property
    def running_services(self):
        return self.list_services()

    @property
    def recent_activities(self):
        return self.list_recent_activities()

    @property
    def active_windows(self):
        return self.list_windows()

    @property
    def focused_window(self) -> t.Optional[WindowRecord]:
        """Gets the currently focused window

        Returns:
            t.Optional[WindowRecord]: The WindowRecord of the topmost window
        """
        cmd = "dumpsys window windows"
        focused = self._list_record_helper(cmd, WindowRecord, pre=".*CurrentFocus=")
        return focused[0] if focused else None

    @property
    def focused_activity(self) -> t.Optional[ActivityRecord]:
        """Gets the currently focused activity

        Returns:
            t.Optional[ActivityRecord]: The ActivityRecord of the topmost activity, None if
                no activity is focused, for example when the screen is locked.
        """
        cmd = "dumpsys window windows"
        focused = self._list_record_helper(cmd, ActivityRecord, pre=".*FocusedApp=.*")
        return focused[0] if focused else None

    def get_time(self):
        """Gets the current time from the device

        Returns:
            datetime: now
        """
        cmd = "date +'%Y-%m-%d %T'"
        return datetime.strptime(self.backend.run(cmd).stdout, "%Y-%m-%d %H:%M:%S")

    def send_tap(self, x: int, y: int) -> bool:
        """Sends a tap to the platform

        Args:
            x (int): x-coordinate
            y (int): y-coordinate

        Returns:
            bool: Command status

        Raises:
            TypeError: Wrong type supplied
        """
        cmd = f"input tap {x} {y}"
        return self.backend.run(cmd).ok

    def send_swipe(self, start: list, end: list, duration: int) -> bool:
        """Sends a keycode/keyevent to the platform

        Args:
            start (list[int]): Start coordinates
            end   (list[int]): End coordinates
            duration (int):    Lenght of the swipe in ms

        Returns:
            bool: Command status

        Raises:
            TypeError: Wrong type supplied
        """
        for inputlist in [start, end]:
            types = set(type(x) for x in inputlist)
            if not all(issubclass(_, (int, np.integer)) for _ in types):
                raise ValueError(f"coordinates should only be int's, but were: {types}")
        cmd = f"input swipe {start[0]} {start[1]} {end[0]} {end[1]} {duration}"
        return self.backend.run(cmd).ok

    def send_keyevent(self, keycode: t.Union[str, int]) -> bool:
        """Sends a keycode/keyevent to the platform

        Args:
            keycode (t.Union[str, int]): The keycode

        Returns:
            bool: Command status

        Raises:
            TypeError: Wrong type supplied
            ValueError: Whitespace in keycode
        """
        if isinstance(keycode, int):
            keycode = str(keycode)
        if not isinstance(keycode, str):
            raise TypeError("keycode", str, type(keycode))
        if re.findall(r"\s", keycode):
            raise ValueError("no whitespace allowed in 'keycode'")

        cmd = "input keyevent {}".format(keycode)
        return self.backend.run(cmd).ok

    def open_home_screen(self) -> bool:
        """Opens the home screen

        Returns:
            bool: Is at home screen?
        """
        _ = self.send_keyevent(KEYCODES["HOME"])
        return self.is_home_screen

    def open_app_switcher(self) -> bool:
        """Opens the app switcher

        Returns:
            bool: Is app switcher opened?
        """
        self.send_keyevent(KEYCODES["APP_SWITCH"])
        return self.is_app_switcher_opened

    @property
    def is_app_switcher_opened(self) -> bool:
        """It the topmost window the app switcher?

        Returns:
            bool: True if focused on the app switcher, False otherwise
        """
        try:
            return "Recents" in self.focused_window.component if self.focused_window else False
        except:
            return False

    def clear_app_switcher(self) -> None:
        """Clears the activities in the app switcher

        Raises:
            RuntimeError: App switcher was not opened

        Returns:
            None: Returns early if only two recent activities exist
        """

        # There will most commonly be at least two activities present: the home screen, and
        # the recents activity. If no others are listed, exit.
        if len(self.list_recent_activities()) == 2:
            return

        if not self.is_screen_on:
            self.turn_screen_on()

        if not self.is_home_screen:
            self.open_home_screen()
        self.open_app_switcher()

        if not self.is_app_switcher_opened:
            raise RuntimeError("app switcher not opened")

        limit = 25
        while self.is_app_switcher_opened and limit > 0:
            self.send_keyevent(KEYCODES["DPAD_DOWN"])
            self.send_keyevent(KEYCODES["DEL"])
            limit -= 1

            if len(self.list_recent_activities()) == 2:
                break
            sleep(0.5)

        self.open_home_screen()

    @property
    def is_home_screen(self) -> bool:
        """Is focused on the home screen?

        Returns:
            bool: True if on the home screen, False otherwise.
        """
        return (
            re.match(
                r".*\.(launcher|homeactivity).*", self.focused_window.component, re.IGNORECASE
            )
            is not None
        )

    @property
    def power_state(self) -> dict:
        """Gets the power state

        Returns:
            dict: The power state
        """
        state = dict.fromkeys(
            [
                "Wakefulness",
                "Interactive",
                "IsPowered",
                "BatteryLevel",
                "SystemReady",
                "StayOn",
                "LowPowerModeEnabled",
                "DisplayReady",
            ]
        )
        dump = self.backend.run("dumpsys power").stdout.strip()

        for key in state:
            _ = re.findall(r"^\s*m" + key + r"=(\S+)", dump, re.IGNORECASE | re.MULTILINE)

            if _:
                value = _[0]

                if re.match(r"true|false", value, re.IGNORECASE):
                    state[key] = True if value == "true" else False
                elif re.match(r"\d+", value, re.IGNORECASE):
                    state[key] = int(value)
                else:
                    state[key] = value

        return state

    def _set_screen_state(self, value: t.Union[bool, str]) -> None:
        """Sets the screen state

        Args:
            value (t.Union[bool, str]): "on" | True or "~" | False

        Raises:
            TypeError: Wrong type supplied as value
        """
        if isinstance(value, str):
            value = True if value.lower() == "on" else False
        if not isinstance(value, bool):
            raise TypeError("value", type(value), bool)

        if value and not self.is_screen_on:
            self.send_keyevent(KEYCODES["POWER"])
        if not value and self.is_screen_on:
            self.send_keyevent(KEYCODES["POWER"])

    def turn_screen_on(self) -> None:
        """Turns the screen on
        """
        self._set_screen_state(True)

    def turn_screen_off(self) -> None:
        """Turns the screen off
        """
        self._set_screen_state(False)

    @property
    def is_screen_on(self) -> bool:
        """Is the screen on?

        Returns:
            bool: True if on, False otherwise.
        """
        return "ON" in self.command("dumpsys nfc | grep mScreenState=").stdout

    @property
    def is_screen_off(self) -> bool:
        """Is screen off?

        Returns:
            bool: True if off, False otherwise.
        """
        return "OFF" in self.command("dumpsys nfc | grep mScreenState=").stdout

    @property
    def is_screen_locked(self) -> bool:
        """Is screen locked?

        Returns:
            bool: True if locked, False otherwise.
        """

        return not "UNLOCKED" in self.command("dumpsys nfc | grep mScreenState=").stdout

    def lock_screen(self) -> None:
        """Locks the screen
        """
        if self.is_screen_on or not self.is_screen_locked:
            self.turn_screen_off()

    def unlock_screen(self) -> None:
        """Unlocks the screen
        """
        sleep(1)

        if self.is_screen_off:
            self.send_keyevent(KEYCODES["POWER"])
            sleep(1)

        if self.is_screen_locked:
            self.send_swipe(start=[300, 1500], end=[900, 900], duration=250)
            sleep(1)

        self.open_home_screen()

    def clear_chrome_tabs(self) -> bool:
        """Removes the temporary data that saves open tabs.

        Returns:
           bool: Command status
        """
        return self.backend.sudo("rm /data/data/com.android.chrome/app_tabs/*/tab*")

    def force_stop_package(self, package: str) -> bool:
        """Forces all activities and tasks of a package to stop

        Args:
            package (str): The package

        Returns:
            bool: Command status

        Raises:
            TypeError: Wrong type supplied for package
        """
        if not isinstance(package, str):
            raise TypeError("package", type(package), str)

        return self.backend.run("am force-stop {}".format(package)).ok

    def kill_activity(self, activity: t.Union[str, ActivityRecord]) -> bool:
        """Killsan activity

        Args:
            activity (t.Union[str, ActivityRecord]): ActivityRecord or activity's component

        Returns:
            bool: Command status

        Raises:
            TypeError: Wrong type supplied for activity
        """
        if not isinstance(activity, (str, ActivityRecord)):
            raise TypeError("activity", type(activity), (str, ActivityRecord))

        current_activities = self.list_activities()

        if isinstance(activity, str):
            _ = list(filter(lambda x: activity in x.component, current_activities))
            if _:
                activity = _[0]
            else:
                return False

        if activity not in current_activities:
            return False

        return self.force_stop_package(activity.component.split("/")[0])

    def kill_activities(self, *activities: t.Union[str, ActivityRecord]) -> None:
        """Kills multiple activities

        Args:
            *activities (t.Union[str, ActivityRecord]): The records or components
        """
        for activity in activities:
            self.kill_activity(activity)

    def send_intent(self, value: Intent, *args, **kwargs) -> ReturnT:
        """Sends an intent

        Args:
            value (Intent): The intent
            *args: Positional arguments to send_intent
            **kwargs: Keyword arguments to send_intent

        Returns:
            ReturnT: The command result

        Raises:
            RuntimeError: Failed to send the intent
        """
        _ = self.backend.send_intent(value, *args, **kwargs)
        if not _.ok and kwargs.get("throw", False):
            raise RuntimeError(f"Sending of intent failed with message: {_.stderr}")

        return _

    def send_intent_via_proxy(self, value: Intent, kind="broadcast", **kwargs) -> ReturnT:
        """Sends an intent via the IntentProxy

        Args:
            value (Intent): The intent
            kind (str, optional): Kind ("service", "broadcast", or proxy-specific actions)
            **kwargs: Keyword arguments to send_intent

        Returns:
            ReturnT: The command result

        Raises:
            RuntimeError: Failed to send the intent
        """
        new_intent = self.intent_to_proxy_intent(value, kind)
        _ = self.backend.send_intent(new_intent, **kwargs)
        if not _.ok and kwargs.get("throw", False):
            raise RuntimeError(f"Sending of intent via proxy failed with message: {_.stderr}")

        return _

    @staticmethod
    def intent_to_proxy_intent(intent: Intent, kind="service") -> Intent:
        """Translates an intent into a IntentProxy's intent

        Args:
            intent (Intent): The intent to translate
            kind (str, optional): Kind ("service", "broadcast", or proxy-specific actions)

        Returns:
            Intent: The translated intent

        Raises:
            ValueError: Kind was not in available keys
        """
        if kind not in ["service", "activity", "broadcast", *INTENTPROXY_ACTIONS]:
            raise ValueError("'kind' must be either service, activity, or broadcast")

        kind_to_action = {
            "service": INTENTPROXY_ACTIONS["forward_startservice"],
            "activity": INTENTPROXY_ACTIONS["forward_startactivity"],
            "broadcast": INTENTPROXY_ACTIONS["forward"],
        }

        kind_to_action.update(**INTENTPROXY_ACTIONS)

        component = intent.COMPONENT_NAME
        action = intent.ACTION
        new_intent = copy(intent)

        new_intent.es.update({"intent.component": component, "intent.action": action})
        new_intent.COMPONENT_NAME = INTENTPROXY_COMPONENT
        new_intent.ACTION = kind_to_action[kind]

        return new_intent

    def start_proxy_service(self) -> bool:
        """Starts the intent proxy service

        Returns:
            bool: True if active, False otherwise.
        """
        self.send_intent(Intent(COMPONENT_NAME=INTENTPROXY_COMPONENT))
        return self.is_proxy_service_active

    def stop_proxy_service(self) -> bool:
        """Stops the intent proxy service

        Returns:
            bool: Command status
        """
        return self.force_stop_package(INTENTPROXY_COMPONENT.split("/")[0])

    @property
    def is_proxy_service_active(self) -> bool:
        """Is the intent proxy service active?

        Returns:
            bool: True if active, False otherwise.
        """
        return bool(list(filter(lambda x: INTENTPROXY_COMPONENT in x, self.list_services())))

    def clear_logs(self) -> None:
        """Clears logs on the device

        Returns:
            bool: Operation was successful?
        """
        self.backend._adb_device_wrapper(["logcat", "-c"])

    def get_logs(
        self,
        grep: t.Optional[str] = None,
        since: t.Optional[t.Union[datetime, str, int]] = None,
        till: t.Optional[t.Union[datetime, str]] = None,
        count: t.Optional[int] = None,
    ) -> t.List[str]:
        """Gets the logs, optionally since a timepoint and with regex matching.
        As the -t option of logcat does not properly work for all android flavors (e.g. Samsung),
        we need to use a workaround to get the log from a certain time on.

        The till option does only work on a second granularity.

        Args:
            grep (t.Optional[str], optional): The regular expression to match
            since (t.Optional[t.Union[datetime, str, int]], optional): The starting timepoint or line count
            till (t.Optional[t.Union[datetime, str]], optional): The end timepoint

        Returns:
            t.List[str]: List of log lines

        Raises:
            TypeError: Wrong type supplied to either argument
        """

        if grep is not None and not isinstance(grep, str):
            raise TypeError("grep", str, type(grep))
        if since is not None and not isinstance(since, (str, datetime, int)):
            raise TypeError("since", (datetime, str, int), type(since))
        if till is not None and not isinstance(till, (str, datetime)):
            raise TypeError("till", (datetime, str), type(till))

        if isinstance(since, datetime):
            logs = ""
            count = 500
            since_search_str = since.strftime("%m-%d %H:%M:%S")
            while since_search_str not in logs and count <= 8500:
                since_time = ["-t", str(count)]
                logs = self.backend._adb_device_wrapper(
                    ["logcat", "-b", "main", "-d", *since_time]
                ).stdout
                count += 500
            if since_search_str not in logs:
                logs = logs.split(since_search_str[:-3], 1)[
                    1
                ]  # Split only on a minute granularity
            else:
                logs = logs.split(since_search_str, 1)[1]
        else:
            since_time = (
                [
                    "-t",
                    since.strftime("%m-%d %H:%M:%S.%f")
                    if isinstance(since, datetime)
                    else str(since),
                ]
                if since
                else []
            )
            logs = self.backend._adb_device_wrapper(
                ["logcat", "-b", "main", "-d", *since_time]
            ).stdout

        if till:
            if till.strftime("%m-%d %H:%M:%S") not in logs:
                logs = logs.split(till.strftime("%m-%d %H:%M"), 1)[0]
            else:
                logs = logs.split(till.strftime("%m-%d %H:%M:%S"), 1)[0]

        return (
            re.findall(r".*" + grep + r".*", logs, re.MULTILINE) if grep else logs.splitlines()
        )
