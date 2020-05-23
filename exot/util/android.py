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
import dataclasses
import re
import typing as t
from collections import namedtuple
from shlex import quote

from invoke.runners import Result as ReturnT

from .misc import list2cmdline

__all__ = (
    "KEYCODES",
    "RECORDS",
    "Intent",
    "RecordParser",
    "TaskRecord",
    "ActivityRecord",
    "ServiceRecord",
    "ProcessRecord",
    "BroadcastRecord",
    "PendingIntentRecord",
    "ContentProviderRecord",
    "WindowRecord",
)

CommandT = t.Union[t.List[str], str]

KEYCODES = dict(
    MENU="KEYCODE_MENU",
    APP_SWITCH="KEYCODE_APP_SWITCH",
    DEL="KEYCODE_DEL",
    BACK="KEYCODE_BACK",
    HOME="KEYCODE_HOME",
    POWER="KEYCODE_POWER",
    DPAD_CENTER="KEYCODE_DPAD_CENTER",
    DPAD_DOWN="KEYCODE_DPAD_DOWN",
    DPAD_DOWN_LEFT="KEYCODE_DPAD_DOWN_LEFT",
    DPAD_DOWN_RIGHT="KEYCODE_DPAD_DOWN_RIGHT",
    DPAD_LEFT="KEYCODE_DPAD_LEFT",
    DPAD_RIGHT="KEYCODE_DPAD_RIGHT",
    DPAD_UP="KEYCODE_DPAD_UP",
    DPAD_UP_LEFT="KEYCODE_DPAD_UP_LEFT",
    DPAD_UP_RIGHT="KEYCODE_DPAD_UP_RIGHT",
)


@dataclasses.dataclass
class Intent:

    """Class for creating Android Intents
    """

    ACTION: str = None
    DATA_URI: str = None
    MIME_TYPE: str = None

    CATEGORY: t.List[str] = dataclasses.field(default_factory=list)
    COMPONENT_NAME: str = None

    es: t.Dict[str, str] = dataclasses.field(default_factory=dict)
    ez: t.Dict[str, bool] = dataclasses.field(default_factory=dict)
    ei: t.Dict[str, int] = dataclasses.field(default_factory=dict)
    el: t.Dict[str, int] = dataclasses.field(default_factory=dict)
    ef: t.Dict[str, float] = dataclasses.field(default_factory=dict)
    eu: t.Dict[str, str] = dataclasses.field(default_factory=dict)
    ecn: t.Dict[str, str] = dataclasses.field(default_factory=dict)
    eia: t.Dict[str, t.List[int]] = dataclasses.field(default_factory=dict)
    eial: t.Dict[str, t.List[int]] = dataclasses.field(default_factory=dict)
    ela: t.Dict[str, t.List[int]] = dataclasses.field(default_factory=dict)
    elal: t.Dict[str, t.List[int]] = dataclasses.field(default_factory=dict)
    efa: t.Dict[str, t.List[float]] = dataclasses.field(default_factory=dict)
    efal: t.Dict[str, t.List[float]] = dataclasses.field(default_factory=dict)
    esa: t.Dict[str, t.List[str]] = dataclasses.field(default_factory=dict)
    esal: t.Dict[str, t.List[str]] = dataclasses.field(default_factory=dict)

    URI: str = None
    PACKAGE: str = None
    COMPONENT: str = None

    def __str__(self) -> str:
        return list2cmdline(self.assemble())

    def assemble(self) -> t.List[str]:
        """Assembles the Intent into a command-line argument sequence for the ActivityManager

        Raises:
            ValueError: If more than one of [URI, PACKAGE, COMPONENT] is provided
            TypeError: If an incompatible type is encountered during annotation-backed parsing

        Returns:
            t.List[str]: A sequence of command-line arguments
        """
        holder = []

        def helper(flag, value, *, do_quote: bool = False):
            if not flag:
                holder.append(value)
                return

            _quote = quote if do_quote else (lambda x: x)

            s_flag = flag.strip("-")

            flag_to_data = {
                "a": "ACTION",
                "d": "DATA_URI",
                "m": "MIME_TYPE",
                "c": "CATEGORY",
                "n": "COMPONENT_NAME",
            }

            _flag = (
                s_flag
                if s_flag not in flag_to_data
                else "{} (-{})".format(flag_to_data[s_flag], s_flag)
            )

            def check_key_types(key_t, k):
                if not isinstance(k, key_t):
                    raise TypeError(
                        f"Flag {_flag!r}: key type was {type(k)} " f"({k!r}), expected {key_t}"
                    )

            def check_value_types(value_t, v):
                if not isinstance(v, value_t):
                    raise TypeError(
                        f"Flag {_flag!r}: value type was {type(v)} "
                        f"({v!r}), expected {value_t}"
                    )

            def check_member_types(member_t, v):
                wrong_elements = [
                    {"index": i, "value": _, "type": type(_)}
                    for i, _ in enumerate(v)
                    if not isinstance(_, member_t)
                ]
                if wrong_elements:
                    raise TypeError(
                        f"Flag {_flag!r}: list expected all elements "
                        f"to have type {member_t}, got wrong elements: "
                        f"{wrong_elements}"
                    )

            if hasattr(self, s_flag) and s_flag in self.__annotations__:
                annotation = self.__annotations__[s_flag]

                if hasattr(annotation, "__origin__"):
                    if annotation.__origin__ is dict:
                        if not isinstance(value, dict):
                            raise TypeError(
                                "Encountered an incompatible instance type", type(value)
                            )

                        is_typing = hasattr(annotation.__args__[1], "__origin__")
                        is_aggregate = hasattr(annotation.__args__[1], "__args__")

                        if is_typing and is_aggregate:
                            key_t, _value_t = annotation.__args__
                            value_t, member_t = _value_t.__origin__, _value_t.__args__[0]

                            for k, v in value.items():
                                check_key_types(key_t, k)
                                check_value_types(value_t, v)
                                check_member_types(member_t, v)

                                if annotation.__args__[1].__args__ == (str,):
                                    v = [_.replace(",", "\,") if "," in _ else _ for _ in v]
                                    holder.extend(
                                        [flag, k, ",".join([_quote(str(_)) for _ in v])]
                                    )
                                else:
                                    holder.extend(
                                        [flag, k, ",".join([_quote(str(_)) for _ in v])]
                                    )
                        else:  # not (is_typing and is_aggregate)
                            for k, v in value.items():
                                key_t, value_t = annotation.__args__
                                check_key_types(key_t, k)
                                check_value_types(value_t, v)

                                holder.extend([flag, k, _quote(str(v))])
                    elif annotation.__origin__ is list:
                        if not isinstance(value, list):
                            raise TypeError(
                                "Encountered an incompatible instance type", type(value)
                            )

                        member_t = annotation.__args__[0]
                        check_member_types(member_t, value)

                        for v in value:
                            holder.extend([flag, _quote(str(value))])
                else:  # not hasattr(annotation, "__origin__")
                    if value:
                        if isinstance(value, list):
                            for v in value:
                                holder.extend([flag, _quote(str(v))])
                        else:
                            holder.extend([flag, _quote(str(value))])
            else:  # not (hasattr(self, s_flag) and s_flag in self.__annotations__)
                if value:
                    if flag_to_data.get(s_flag, None) in self.__annotations__:
                        annotation = self.__annotations__[flag_to_data[s_flag]]
                        key_t = (
                            annotation
                            if not hasattr(annotation, "__origin__")
                            else annotation.__origin__
                        )
                        member_t = getattr(annotation, "__args__", [None])[0]

                        check_key_types(key_t, value)
                        if member_t:
                            check_member_types(member_t, value)

                    if isinstance(value, list):
                        for v in value:
                            holder.extend([flag, _quote(str(v))])
                    else:
                        holder.extend([flag, _quote(str(value))])

        helper("-a", self.ACTION)
        helper("-d", self.DATA_URI)
        helper("-t", self.MIME_TYPE)
        helper("-c", self.CATEGORY)
        helper("-n", self.COMPONENT_NAME)

        helper("--es", self.es, do_quote=True)
        helper("--ez", self.ez)
        helper("--ei", self.ei)
        helper("--el", self.el)
        helper("--ef", self.ef)
        helper("--eu", self.eu)
        helper("--ecn", self.ecn)
        helper("--eia", self.eia)
        helper("--eial", self.eial)
        helper("--ela", self.ela)
        helper("--elal", self.elal)
        helper("--efa", self.efa)
        helper("--efal", self.efal)
        helper("--esa", self.esa, do_quote=True)
        helper("--esal", self.esal, do_quote=True)

        one_of_upc = [_ for _ in [self.URI, self.PACKAGE, self.COMPONENT] if _ is not None]

        if len(one_of_upc) > 1:
            raise ValueError(
                "Only one of [URI, PACKAGE, COMPONENT] must be provided to the Intent"
            )
        elif one_of_upc:
            helper(None, *one_of_upc)

        return holder

    def __bool__(self) -> bool:
        """Checks if the Intent has any field specified

        Returns:
            bool: True if at least one field is specified
        """
        return any(dataclasses.astuple(self))

    def to_uri(self, runner: t.Callable[[CommandT], ReturnT]) -> str:
        """Converts the intent to an URI

        Args:
            runner (t.Callable[[t.List[str]], Backend.ReturnT]):
                A callable that runs shell commands on the target

        Returns:
            str: The URI

        Raises:
            RuntimeError: If the command failed
        """
        request = ["am", "to-uri"]
        request.extend(self.assemble())

        _ = runner(request)
        if _.exited == 0:
            return _.stdout
        else:
            raise RuntimeError(_.stderr)

    def to_intent_uri(self, runner: t.Callable[[CommandT], ReturnT]) -> str:
        """Converts the intent to an Intent URI

        Args:
            runner (t.Callable[[t.List[str]], Backend.ReturnT]):
                A callable that runs shell commands on the target

        Returns:
            str: The Intent URI

        Raises:
            RuntimeError: If the command failed
        """
        request = ["am", "to-intent-uri"]
        request.extend(self.assemble())

        _ = runner(request)
        if _.exited == 0:
            return _.stdout
        else:
            raise RuntimeError(_.stderr)

    def to_app_uri(self, runner: t.Callable[[CommandT], ReturnT]) -> str:
        """Converts the intent to an app URI

        Args:
            runner (t.Callable[[t.List[str]], Backend.ReturnT]):
                A callable that runs shell commands on the target

        Returns:
            str: The App URI

        Raises:
            RuntimeError: If the command failed
        """
        request = ["am", "to-app-uri"]
        request.extend(self.assemble())

        _ = runner(request)
        if _.exited == 0:
            return _.stdout
        else:
            raise RuntimeError(_.stderr)


# fmt: off
# TaskRecord{111a544 #819 A=com.android.systemui U=0 StackId=0 sz=1}
TaskRecord = namedtuple("TaskRecord",
    ["hash", "taskId", "component", "userId", "stackId", "activitiesCount"])
# ActivityRecord{700d205 u0 com.sonyericsson.home/com.sonymobile.home.HomeActivity t816}
ActivityRecord = namedtuple("ActivityRecord",
    ["hash", "userId", "component", "taskId", "stackId"], defaults=[None])
# ServiceRecord{27a78e9 u0 com.google.android.gms/.common.stats.GmsCoreStatsService}
ServiceRecord = namedtuple("ServiceRecord",
    ["hash", "userId", "component"])
# ProcessRecord{95d2c79 4182:com.google.android.gms.persistent/u0a52}
ProcessRecord = namedtuple("ProcessRecord",
    ["hash", "pid", "process", "userId", "userAppId", "appId", "uId"], defaults=[None, None, None, None])
# BroadcastRecord{5f3c483 u-1 android.intent.action.TIME_TICK}
BroadcastRecord = namedtuple("BroadcastRecord",
    ["hash", "userId", "action"])
# PendingIntentRecord{d058f7c android broadcastIntent}
PendingIntentRecord = namedtuple("PendingIntentRecord",
    ["hash", "package", "type"])
# ContentProviderRecord{dff7021 u0 com.android.providers.telephony/.TelephonyProvider}
ContentProviderRecord = namedtuple("ContentProviderRecord",
    ["hash", "userId", "component"])
# Window{5a027a8 u0 com.cygery.repetitouch.pro}
WindowRecord = namedtuple("WindowRecord",
    ["hash", "userId", "component"])
# fmt: on


RECORDS = {
    "TaskRecord": TaskRecord,
    "ActivityRecord": ActivityRecord,
    "ServiceRecord": ServiceRecord,
    "ProcessRecord": ProcessRecord,
    "BroadcastRecord": BroadcastRecord,
    "PendingIntentRecord": PendingIntentRecord,
    "ContentProviderRecord": ContentProviderRecord,
    "WindowRecord": WindowRecord,
}


class RecordParser:
    RecordRegex = {
        TaskRecord: r"TaskRecord\{([a-zA-Z\d]+) #(\d+) a?[AI]=([a-zA-Z\d./]+) U=(-?\d+)\s*(?:StackId=(\d+))* sz=(\d+)\}",
        ActivityRecord: r"ActivityRecord\{([a-zA-Z\d]+) u(-?\d+) ([a-zA-Z\d./]+) t(\d+)\}",
        ServiceRecord: r"ServiceRecord\{([a-zA-Z\d]+) u(-?\d+) ([a-zA-Z\d./]+)\}",
        ProcessRecord: r"ProcessRecord\{([a-zA-Z\d]+) (\d+):([a-zA-Z\d.]+)/(\d+|u-?\d+)*(a-?\d+)*(s-?\d+)*(i-?\d+)*\}",
        BroadcastRecord: r"BroadcastRecord\{([a-zA-Z\d]+) \}",
        PendingIntentRecord: r"PendingIntentRecord\{([a-zA-Z\d]+) u(-?\d+) (\S+).*\}",
        ContentProviderRecord: r"ContentProviderRecord\{([a-zA-Z\d]+) u(-?\d+) ([a-zA-Z\d./]+)\}",
        WindowRecord: r"Window\{([a-zA-Z\d]+) u(-?\d+) ([a-zA-Z\d./ ]+)\}",
    }

    RecordTypes = [_ for _ in RecordRegex]

    @staticmethod
    def parse_record(
        value: str, record: t.T, *, exact=False, pre=r"", post=r"", **kwargs
    ) -> t.Optional[t.T]:
        """Parses a string into an Android record type

        Args:
            value (str): The string to parse
            record (t.T): Description

        Returns:
            t.Optional[t.T]: Description

        Raises:
            TypeError: Description
        """
        if not isinstance(value, str):
            raise TypeError(type(value), "not", str)
        if not isinstance(exact, bool):
            raise TypeError(type(exact), "not", bool)
        if not isinstance(pre, str):
            raise TypeError(type(pre), "not", str)
        if not isinstance(post, str):
            raise TypeError(type(post), "not", str)
        if record not in RecordParser.RecordRegex:
            raise TypeError(type(record), "not in", RecordParser.RecordRegex.keys())

        regex = (
            pre + RecordParser.RecordRegex[record] + post
            if exact
            else ".*" + pre + RecordParser.RecordRegex[record] + post + ".*"
        )
        match = re.match(regex, value)

        if kwargs.pop("verbose", False):
            print({"regex": regex, "line": value, "match": match})

        if not match:
            return None

        _ = [*match.groups()]

        for arg in kwargs:
            if arg in record._fields:
                idx = record._fields.index(arg)

                try:
                    if idx < len(_):
                        _[idx] = kwargs[arg]
                    else:
                        _.insert(idx, kwargs[arg])
                except IndexError as e:
                    e.args += (idx, arg, kwargs[arg])
                    raise e

        return record(*_)

    @staticmethod
    def parse_any_record(value, **kwargs) -> t.Optional[t.T]:
        for T in RecordParser.RecordRegex:
            _ = RecordParser.parse_record(value, T, **kwargs)
            if _:
                return _
