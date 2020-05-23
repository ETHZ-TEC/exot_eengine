"""TimeValue I/O layer"""

import typing as t
from pathlib import Path

import numpy as np
import pandas as pd

from exot.exceptions import *
from exot.util.mixins import LabelConversions
from exot.util.scinum import get_nearest
from exot.util.wrangle import *

from .._base import Layer

__all__ = ("TimeValue",)


"""
TimeValue
---------

The TimeValue input/output layer works for time-specific schedule-generating encode flows,
and timestamped measurement decoding. It performs the following:

-   [__Encode__]: takes a DataFrame with timings in seconds, and specific schedules as
    subsequent columns, where the column name corresponds to the schedule tag. Timings are
    recalculated to fit the order of magnitude required by the app, e.g. microseconds.
    The timing multiplier is a instantiation and runtime configurable parameter.

    Additionally, a `write_schedules` method is provided, which will write schedules at
    the path given in the configuration.

-   [__Decode__]: takes a DataFrame, recalculates timestamps to seconds and applies a
    Matcher to select desired columns. See `Matcher` in exot.util.wrangle for details.

    Additionally, a `get_measurements` is provided, which reads a measurement given an
    'env' (experiment environment), and a 'rep' (run repetition) in config. Moreover,
    there are methods to get the available logs and environments, `describe_measurements`,
    `get_available_environments`, `get_available_app_logs`, `get_available_debug_logs`.
"""


class TimeValue(Layer, LabelConversions, layer=Layer.Type.InputOutput):
    def __init__(self, *, timebase: str = "ns", synchronise: bool = True, **kwargs):
        """Initialise the TimeValue I/O layer

        Args:
            timebase (str, optional): The timebase, either "s", "ms", "us" or "ns"
        """

        self.timebase = timebase
        self.synchronise = synchronise

    @property
    def _encode_types(self) -> t.Tuple[type]:
        return (pd.DataFrame,)

    @property
    def _decode_types(self) -> t.Tuple[type]:
        return (pd.DataFrame,)

    @property
    def _encode_validators(self):
        return {
            np.ndarray: [
                lambda v: v.ndim == 2,
                lambda v: v.shape[1] >= 2,
                lambda v: "timestamp" in v,
            ]
        }

    @property
    def _decode_validators(self):
        # Accepts: timestamps + single value
        return self._encode_validators

    def validate(self):
        """Validate config

        Implementation of Configurable's `validate` method
        """
        try:
            _ = self.config
            if "env" in _:
                assert isinstance(_.env, str), ("env", str, type(_.env))
            if "app" in _:
                assert isinstance(_.app, str), ("app", str, type(_.app))
            if "rep" in _:
                assert isinstance(_.rep, int), ("rep", int, type(_.rep))
            if "matcher" in _:
                assert isinstance(_.matcher, Matcher), ("Matcher", Matcher, type(_.matcher))
            if "path" in _:
                assert isinstance(_.path, Path), ("path", Path, type(_.path))
        except AssertionError as e:
            raise MisconfiguredError("timevalue: {} expected {}, got: {}".format(*e.args[0]))

    @property
    def requires_runtime_config(self) -> bool:
        """Does encode/decode require runtime configuration?

        Encoding needs to be provided:
        - "path": run path. (for writing schedules)

        Decoding needs to be provided:
        - "env": environment name,
        - "rep": repetition number,
        - "path": run path,
        - "matcher": selection specifiers.
        """
        return (True, True)

    @property
    def schedules(self) -> t.Mapping[str, pd.DataFrame]:
        """Get schedules"""
        return getattr(self, "_schedules", None)

    @schedules.setter
    def schedules(self, value: t.Mapping[str, pd.DataFrame]) -> None:
        """Set schedules"""
        assert isinstance(value, t.Mapping), "must be a dict"
        assert all(isinstance(_, pd.DataFrame) for _ in value.values()), "must have DataFrames"
        self._schedules = value

    @property
    def timebase(self) -> int:
        """Get the timebase as a number of divisions of a second"""
        return self._timebase

    @timebase.setter
    def timebase(self, value: str) -> None:
        """Set the timebase with a string representation"""
        valid_timebases = ["s", "ms", "us", "ns"]

        if not isinstance(value, str):
            raise LayerMisconfigured("the 'timebase' must be a string")
        if value not in valid_timebases:
            raise LayerMisconfigured("the 'timebase' must be one of {!r}".format(value))

        elif value == "ms":
            self._timebase = 1e03
        elif value == "us":
            self._timebase = 1e06
        elif value == "ns":
            self._timebase = 1e09
        else:
            self._timebase = 1

    @property
    def synchronise(self):
        return self._synchronise

    @synchronise.setter
    def synchronise(self, value):
        if not isinstance(value, bool):
            raise LayerMisconfigured("the 'synchronise' option must be a boolean")

        self._synchronise = value

    def _encode(self, rdpstream: pd.DataFrame) -> pd.DataFrame:
        """Encode the rdpstream and produce schedules

        Args:
            rdpstream (pd.DataFrame): the DataFrame produced by a RDP layer

        Returns:
            pd.DataFrame: The rdpstream with platform-adjusted timing
        """
        rawstream = rdpstream.copy()
        rawstream["timestamp"] = (np.round(rawstream["timestamp"] * self.timebase)).astype(int)
        self.schedules = self._make_schedules(rawstream)
        return rawstream

    def _make_schedules(self, rawstream: pd.DataFrame) -> t.Mapping[str, pd.DataFrame]:
        """Make separate schedule DataFrame's

        Args:
            rawstream (pd.DataFrame): the rawstream produced by this layer

        Returns:
            t.Mapping[pd.DataFrame]: A mapping: schedule name -> schedule DataFrame
        """
        _schedules = {}
        _ts = pd.DataFrame(rawstream["timestamp"])

        for column, values in rawstream.drop("timestamp", 1).items():
            _schedules[column] = _ts.join(values)

        return _schedules

    def write_schedules(self, schedules: t.Mapping[str, pd.DataFrame]) -> t.List[Path]:
        """Write schedules to files"""
        assert self.configured, "must be configured"
        assert "path" in self.config, "must have a path"
        paths = []

        for name, frame in schedules.items():
            path_to_schedule = self.config.path / (name + ".sched")
            paths.append(path_to_schedule)
            if not path_to_schedule.parent.is_dir():
                path_to_schedule.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path_to_schedule, index=False, header=False, sep="\t")

        return paths

    def get_measurements(self, app="snk") -> pd.DataFrame:
        """Get measurements for a specific environment, app and repetition

        Returns:
            pd.DataFrame: a DataFrame constructed from a raw CSV app log

        Raises:
            LayerRuntimeError: If an unexpected missing value is encountered
            LayerRuntimeMisconfigured: If a rep or env is not available
        """
        assert self.configured
        assert "env" in self.config, "must provide an env in config"
        assert "rep" in self.config, "must provide a rep in config"
        app = self.config.app if "app" in self.config else app

        return self.get_logs(self.config.env, app, self.config.rep)

    def get_logs(self, env, app, rep, *, suffix=".log.csv", **kwargs):
        all_logs = self.get_available_logs("*/*{}".format(suffix))

        if env not in all_logs:
            raise LayerRuntimeMisconfigured(f"{env!r} not in available envs: {list(all_logs)}")
        if app not in all_logs[env]:
            raise LayerRuntimeMisconfigured(
                f"logs for app {app!r} not available for env: {env!r}"
            )

        files = all_logs[env][app]
        registry = {app_log_unformatter(_)[0]: _ for _ in files}

        if rep not in registry:
            raise LayerRuntimeMisconfigured(
                f"{rep} not in available reps: {list(registry.keys())}"
            )

        if suffix.endswith("csv"):
            return pd.read_csv(registry[rep], **kwargs)
        else:
            with registry[rep].open() as f:
                return f.read()

    def describe_measurements(self, data: pd.DataFrame) -> t.Mapping:
        """Describe the values contained in a raw measurements DataFrame

        Args:
            data (pd.DataFrame): a measurements DataFrame

        Returns:
            t.Mapping: a dict with available parameters
        """
        return parse_header(data.columns)

    def get_available_environments(self) -> t.List[str]:
        """Get the environments which have logs available

        Returns:
            t.List[str]: A list of environment names
        """
        return list(self.get_available_app_logs().keys())

    def get_available_logs(self, glob="*/*.log.csv") -> t.Mapping:
        """Get the available logs at the config path that match the chosen glob pattern

        Returns:
            t.Mapping: a mapping env: str -> app: str -> list of file Paths
        """
        assert self.configured, "must be configured"
        assert "path" in self.config, "must have path in config"
        assert isinstance(glob, str), "'glob' must be a string"

        logs = sorted(list(self.config.path.rglob(glob)))
        envs = {_.parent.name for _ in logs}
        out = {k: {} for k in envs}

        for log in logs:
            rep, app, dbg = app_log_unformatter(log)
            out[log.parent.name][app] = out[log.parent.name].get(app, [])
            out[log.parent.name][app].append(log)

        return out

    def get_available_app_logs(self) -> t.Mapping:
        """Get the available app logs at the config path

        Returns:
            t.Mapping: a mapping env: str -> app: str -> list of file Paths
        """
        return self.get_available_logs("*/*.log.csv")

    def get_available_debug_logs(self) -> t.Mapping:
        """Get the available app logs at the config path

        Returns:
            t.Mapping: a mapping env: str -> app: str -> list of file Paths
        """
        return self.get_available_logs("*/*.debug.txt")

    def _decode(self, rawstream: pd.DataFrame) -> pd.DataFrame:
        """Decode an input rawstream and produce a RDP-compatible stream

        Args:
            rawstream (pd.DataFrame): a rawstream, produced by `get_measurements`

        Returns:
            pd.DataFrame: a filtered DataFrame with adjusted timing values
        """
        assert self.configured, "must be configured"
        assert "matcher" in self.config, "must have a matcher in config"

        if (len(self.config.matcher._dimensions) > 1) and ("combine" in self.config):
            if self.config.combine == "max":
                tmp_data = rawstream[self.config.matcher].iloc[:, 1:].max(axis=1).copy()
            elif self.config.combine == "min":
                tmp_data = rawstream[self.config.matcher].iloc[:, 1:].min(axis=1).copy()
            else:
                tmp_data = rawstream[self.config.matcher].iloc[:, 1:].sum(axis=1).copy()
            columnnames = rawstream[self.config.matcher].columns
            rdpstream = pd.DataFrame(
                {
                    columnnames[0]: rawstream[self.config.matcher].iloc[:, 0].copy(),
                    columnnames[1].replace(
                        self.config.matcher._dimensions[0], self.config.combine
                    ): tmp_data,
                }
            )

        else:
            rdpstream = rawstream[self.config.matcher].copy(deep=True)

        try:
            _ = self.get_logs(self.config.env, "src", self.config.rep)
            self.add_intermediate("raw_src_log", _)
        except (LayerRuntimeMisconfigured, Exception):
            pass

        if self.synchronise:
            if "raw_src_log" not in self.intermediates:
                raise LayerRuntimeError(
                    "'synchronise' was set to True but a matching 'src' log was not found"
                )

            # slice the rdpstream
            idx, val = get_nearest(
                rdpstream.iloc[:, 0], self.intermediates.raw_src_log.iloc[0, 0]
            )

            diff = rdpstream.iloc[idx, 0] - self.intermediates.raw_src_log.iloc[0, 0]
            streamdiff = rdpstream.iloc[:, 0].diff().dropna().mean() / self.timebase
            idx = max(0, idx - 1) if (diff / self.timebase) > (streamdiff * 0.5) else idx

            rdpstream = rdpstream.iloc[idx:, :]

            # normalise the raw src log
            _ = self.intermediates.raw_src_log.copy(deep=True)
            _.iloc[:, 0] = _.iloc[:, 0].div(self.timebase)
            _.iloc[:, 0] = _.iloc[:, 0] - (rdpstream.iloc[0, 0] / self.timebase)
            _.iloc[:, 1] = _.iloc[:, 1].div(self.timebase)
            _.iloc[:, 1] = _.iloc[:, 1] - (rdpstream.iloc[0, 0] / self.timebase)
            _.iloc[:, 2] = _.iloc[:, 2].div(self.timebase)

            _ = _.rename(
                columns={
                    _.keys()[0]: self._label_unit_conversion(_.keys()[0], self.timebase),
                    _.keys()[1]: self._label_unit_conversion(_.keys()[1], self.timebase),
                    _.keys()[2]: self._label_unit_conversion(_.keys()[2], self.timebase),
                }
            )

            self.add_intermediate("src_log", _)

        rdpstream.iloc[:, 0] = rdpstream.iloc[:, 0].div(self.timebase)
        rdpstream.iloc[:, 0] = rdpstream.iloc[:, 0] - rdpstream.iloc[0, 0]
        rdpstream = rdpstream.rename(
            columns={
                rdpstream.keys()[0]: self._label_unit_conversion(
                    rdpstream.keys()[0], self.timebase
                )
            }
        )

        return rdpstream
