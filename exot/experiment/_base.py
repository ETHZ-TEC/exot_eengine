from __future__ import annotations

import abc
import concurrent.futures
import copy as cp
import datetime
import enum
import json
import os
import tarfile
import threading
import traceback
import typing as t
from functools import partial
from pathlib import Path
from time import sleep

import numpy as np
import toml

from exot.channel import Channel
from exot.driver import Driver, DriverFactory
from exot.exceptions import *
from exot.layer import LayerFactory
from exot.util.android import Intent
from exot.util.attributedict import AttributeDict
from exot.util.file import (
    add_random,
    add_timestamp,
    backup as file_backup,
    copy,
    delete,
    move,
    move_action,
    validate_root_path,
)
from exot.util.git import GitRepository
from exot.util.logging import Loggable, get_root_logger
from exot.util.misc import (
    call_with_leaves,
    dict_depth,
    dict_diff,
    get_valid_access_paths,
    getitem,
    isabstract,
    leaves,
    map_to_leaves,
    setitem,
    validate_helper,
)
from exot.util.mixins import (
    Configurable,
    HasParent,
    IntermediatesHandler,
    Pickleable,
    SubclassTracker,
)
from exot.util.prompts import prompt
from exot.util.timeout import Timeout
from exot.util.wrangle import (
    app_log_formatter,
    log_path_unformatter,
    generic_log_formatter,
    run_path_formatter,
    run_path_unformatter,
    app_configfile_formatter,
    repetition_formatter,
)

__all__ = ("Experiment", "Run")


"""
Experiment
----------

Base class for experiments. Responsible for:

- configuration parsing and validation,
- bootstrapping data processing layers, drivers, and channels,
- finding and validating environments,
- writing and reading serialised experiments,
- backing up the experiment.

Synopsis & signatures::

__init_subclass__      (*args, **kwargs) -> None
_create_drivers        (self) -> 't.Mapping'
_get_environments      (self, _ignore: 't.Optional[str]' = None)
_validate_environment  (env: 't.Mapping') -> 't.NoReturn'
_validate_execute      (self, env : str) -> 't.NoReturn'
backup                 (self, _with_logs: 'bool' = False, _with_results: 'bool' = False) -> 'None'
bootstrap              (self) -> 'None'
bootstrap_logger       (self, custom_filename: Union[str, NoneType] = None) -> None
configure_layers       (self, **kwargs: 't.Any') -> 'None'
deserialise            (source=typing.Union[str, bytes, pathlib.Path, typing.IO]) -> object
dump                   (self, file: io.IOBase) -> None
dumps                  (self) -> Union[str, bytes]
execute                (self) -> 'None'
generate               (self) -> 'None'
load                   (source: io.IOBase) -> object
loads                  (source: Union[str, bytes]) -> object
read                   (path: 'Path') -> 'Experiment'
required_config_keys   () -> 'list'
required_layers        () -> 'list'
serialise              (self, file: Union[pathlib.Path, IO, NoneType] = None) -> Union[str, NoneType]
validate               (self) -> 'None'
write                  (self) -> 'None'
"""


class Experiment(
    SubclassTracker,
    Pickleable,
    Configurable,
    Loggable,
    track="type",
    serialise_stub=["_phases"],
    metaclass=abc.ABCMeta,
):
    @enum.unique
    class Type(enum.IntEnum):
        """Experiment types"""

        FrequencySweep = enum.auto()
        Exploratory    = enum.auto()
        Performance    = enum.auto()
        RealWorld      = enum.auto()
        AppExec        = enum.auto()

    def __init_subclass__(cls, *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs)

    @property
    def _config_general_standard(self):
        return "STANDARD"

    @property
    def config(self):
        if hasattr(self, '_config'):
            setting_key = self._config_general_standard
            if hasattr(self.layers, 'io'):
                if self.layers['io'].configured:
                    if 'env' in self.layers['io'].config:
                        setting_key = self.layers['io'].config['env']
            if setting_key in self._config.EXPERIMENT.GENERAL:
                elem_keys = list(self._config.EXPERIMENT.GENERAL[setting_key])
                for elem_key in elem_keys:
                    self._config.EXPERIMENT.GENERAL[elem_key] = cp.deepcopy(self._config.EXPERIMENT.GENERAL[setting_key][elem_key])
            return self._config
        else:
            return None

    @config.setter
    def config(self, value):
        self._config = AttributeDict(value)
        elem_keys = list(self._config.EXPERIMENT.GENERAL)
        if self._config_general_standard not in self._config.EXPERIMENT.GENERAL:
            self._config.EXPERIMENT.GENERAL[self._config_general_standard] = AttributeDict()
        for key in elem_keys:
            if key != self._config_general_standard and key not in list(self.environments.keys()):
                self._config.EXPERIMENT.GENERAL[self._config_general_standard][key] = cp.deepcopy(self._config.EXPERIMENT.GENERAL[key])

    @config.deleter
    def config(self):
        if hasattr(self, '_config'):
            delattr(self, '_config')

    @property
    def run_type(self):
        """Get the associated Run type"""
        return self._run_type

    @run_type.setter
    def run_type(self, value):
        """Set the associated Run type"""
        if not (isinstance(value, type) and issubclass(value, Run)):
            raise TypeError("run type should be a subclass of Run")

        self._run_type = value

    def __init__(self, *args, **kwargs):
        """Initialise an Experiment object"""
        # Validate root path
        if "root_path" in kwargs:
            self._root_path = Path(kwargs.pop("root_path"))
        else:
            self._root_path = Path(".")

        validate_root_path(self.root_path)

        if self.root_path != Path("."):
            get_root_logger().warning(
                "creating an experiment with a different root path might complicate "
                "reading back"
            )

        if self.root_path.is_absolute():
            get_root_logger().warning(
                "creating an experiment with an absolute root path might be difficult "
                "to read back by others"
            )

        # Check git status
        if GitRepository.is_git_directory(self.root_path):
            repo = GitRepository(self.root_path)
            if repo.dirty:
                get_root_logger().warning("the repository contains uncommited modifications!")
            self._git = {"commit": repo.commit, "dirty": repo.dirty, "status": repo.status}
        else:
            get_root_logger().warning(
                "root path is not a git directory, inconsistencies may occur!"
            )
            self._git = None

        # Initialise the Configurable parent class
        Configurable.__init__(self, *args, **kwargs)
        # Loggable needs path methods to be available
        Loggable.__init__(self, *args, **kwargs)

        if "channel" in kwargs:
            self.channel = kwargs.pop("channel")
            self.channel.parent = self

        if "drivers" in kwargs:
            self.drivers = kwargs.pop("drivers")

        if "run_type" in kwargs:
            self.run_type = kwargs.pop("run_type")

        self._layers = AttributeDict()

        if self.configured:
            self.bootstrap()

    """
    Bootstrapping
    -------------

    Values that are always provided to layers at instantiation time:
    - Channel,
    - Channel's signal.
    - sampling period.

    Values that are optionally provided to layers at instantiation time:
    - Keyword arguments provided to the `bootstrap` function.
    """

    def bootstrap(self, **kwargs) -> None:
        """Parse configuration and create experiment layers using a layer factory"""
        assert self.configured, "Experiment must be configured before bootstrapping"
        assert self.channel, "Experiment must have a channel before bootstrapping"

        # broadcast channel and channel's signal to all layers
        _env = ""
        for item in kwargs.values():
            if "env" in item.keys():
                _env = item["env"]
                break
        kwargs.update(
            channel=self.channel,
            environments_apps_zones=self.environments_apps_zones,
            sampling_period=self.environment_config_general(_env).sampling_period,
        )

        layer_factory = LayerFactory()
        layer_conf = self.config.EXPERIMENT.LAYERS
        layer_types = {t.value: t for t in layer_factory.available_types}

        for layer_k, layer_v in layer_conf.items():
            try:
                _type = {"_type": layer_types[layer_k]} if layer_k in layer_types else {}
                self.layers[layer_k] = layer_factory(
                    layer_v.name, **_type, **{**layer_v.params, **kwargs}
                )
                self.logger.info(f"bootstrapped layer {layer_k!r} with {layer_v.name}")
            except AttributeError as e:
                self.logger.critical(
                    f"failed to bootstrap layer {layer_k!r} with {layer_v.name}"
                )
                # Handle missing EXPERIMENT.LAYERS.* as misconfiguration
                raise MisconfiguredError(e.args)

        self._runtime_encode = AttributeDict(
            _ for _ in self.layers.items() if _[1].requires_runtime_config[0]
        )

        self._runtime_decode = AttributeDict(
            _ for _ in self.layers.items() if _[1].requires_runtime_config[1]
        )

        self.logger.debug(f"layers with runtime encoding: {list(self._runtime_encode)}")
        self.logger.debug(f"layers with runtime decoding: {list(self._runtime_decode)}")
        self._bootstrapped = True

    @property
    def bootstrapped(self) -> bool:
        """Are the experiment layers bootstrapped?"""
        return getattr(self, "_bootstrapped", False)

    @property
    def layers(self) -> AttributeDict:
        """Get experiment layers"""
        return getattr(self, '_layers', None)

    @property
    def layers_with_runtime_encoding(self) -> AttributeDict:
        """Get experiment layers that require configuration at runtime"""
        return getattr(self, "_runtime_encode", AttributeDict())

    @property
    def layers_with_runtime_decoding(self) -> AttributeDict:
        """Get experiment layers that require configuration at runtime"""
        return getattr(self, "_runtime_decode", AttributeDict())

    def configure_layers_encoding(self, **kwargs: t.Any) -> None:
        """Configure layers with runtime-configurable encoding"""
        for layer in self.layers_with_runtime_encoding:
            if layer in kwargs:
                self.layers_with_runtime_encoding[layer].config = kwargs.get(layer, {})
            else:
                self.logger.debug("layer {layer!r} encoding not runtime-configurable")

    def configure_layers_decoding(self, **kwargs: t.Any) -> None:
        """Configure layers with runtime-configurable decoding"""
        for layer in self.layers_with_runtime_decoding:
            if layer in kwargs:
                self.layers_with_runtime_decoding[layer].config = kwargs.get(layer, {})
            else:
                self.logger.debug("layer {layer!r} decoding not runtime-configurable")

    def __repr__(self) -> str:
        """Get a string representation of the Experiment"""
        channel = getattr(self, "_channel", None)
        channel = channel.__class__.__name__ if channel else "no"
        configured = "configured" if self.configured else "not configured"

        return (
            f"<{self.__class__.__name__} at {hex(id(self))} "
            f"with {channel} channel, {configured}>"
        )

    @property
    def required_config_keys(self) -> list:
        """Gets the required config keys

        Returns:
            list: The required keys
        """
        return ["name", "save_path", "EXPERIMENT", "ENVIRONMENTS"]

    @staticmethod
    def required_layers() -> list:
        """Which layers are required? Used in verification/validation"""
        return []

    @property
    def estimated_delays_duration(self) -> float:
        """Gets the estimated duration of the defined delays

        Returns:
            float: The estimated delay in seconds
        """
        if self.configured:
            return (
                self.config.EXPERIMENT.GENERAL.delay_after_spawn
                if "delay_after_spawn" in self.config.EXPERIMENT.GENERAL
                else 1.0
            ) + (
                self.config.EXPERIMENT.GENERAL.delay_after_auxiliary
                if "delay_after_auxiliary" in self.config.EXPERIMENT.GENERAL
                else 1.0
            )
        else:
            return 0.0

    def validate(self) -> None:
        """Check if the supplied Experiment config is valid

        Implements the `validate` method from Configurable.
        """

        if not self.configured:
            raise ConfigMissingError("'validate' called on an unconfigured Experiment")

        validate = partial(validate_helper, self.config, msg="Experiment")

        # top-level configuration
        validate("name", str)
        validate("save_path", str)
        validate("log_path", str, type(None))
        validate("backup_path", str, type(None))
        validate("experiment_exists_action", str, type(None))
        validate("EXPERIMENT", AttributeDict)
        validate("ENVIRONMENTS", AttributeDict)

        validate(("EXPERIMENT", "GENERAL"), AttributeDict)
        validate(("EXPERIMENT", "GENERAL", "timeout_duration"), (int, float, type(None)))
        validate(("EXPERIMENT", "GENERAL", "delay_after_spawn"), (int, float, type(None)))
        validate(("EXPERIMENT", "GENERAL", "delay_after_auxiliary"), (int, float, type(None)))
        validate(("EXPERIMENT", "GENERAL", "active_wait"), (bool, type(None)))
        validate(("EXPERIMENT", "GENERAL", "stagger"), (bool, type(None)))

        # environments configuration
        for k in self.config["ENVIRONMENTS"]:
            validate(("ENVIRONMENTS", k), AttributeDict)
            validate(("ENVIRONMENTS", k, "APPS"), AttributeDict)

            for app in self.config["ENVIRONMENTS"][k]["APPS"]:
                validate(("ENVIRONMENTS", k, "APPS", app, "executable"), str)
                validate(("ENVIRONMENTS", k, "APPS", app, "zone"), str)
                validate(("ENVIRONMENTS", k, "APPS", app, "type"), str, type(None))
                validate(
                    ("ENVIRONMENTS", k, "APPS", app, "start_individually"), bool, type(None)
                )

                current_app_config = self.config["ENVIRONMENTS"][k]["APPS"][app]

                if "type" in current_app_config:
                    if current_app_config["type"] not in ["local", "standalone"]:
                        raise MisconfiguredError(
                            f"App {app!r} in section 'APPS' of environment {k!r} "
                            f"has incompatible type: {current_app_config['type']!r}, "
                            f"allowed types: ['local', 'standalone']."
                        )

                    if current_app_config["type"] == "local":
                        validate(("ENVIRONMENTS", k, "APPS", app, "sched"), str, type(None))
                    else:
                        validate(("ENVIRONMENTS", k, "APPS", app, "args"), list, type(None))

        # experiment configuration
        validate(("EXPERIMENT", "type"), str)
        validate(("EXPERIMENT", "channel"), str)
        validate(("EXPERIMENT", "PHASES"), AttributeDict)
        validate(("EXPERIMENT", "LAYERS"), AttributeDict)

        missing_layers = [
            layer
            for layer in self.required_layers()
            if layer not in self.config["EXPERIMENT"]["LAYERS"]
        ]

        if missing_layers:
            raise MisconfiguredError(f"required layers missing: {missing_layers!r}")

        for k in self.config["EXPERIMENT"]["LAYERS"]:
            validate(("EXPERIMENT", "LAYERS", k), AttributeDict)
            validate(("EXPERIMENT", "LAYERS", k, "name"), str)
            validate(("EXPERIMENT", "LAYERS", k, "params"), t.Mapping)

        # backup configuration
        if "BACKUP" in self.config:
            for k in self.config["BACKUP"]:
                validate(("BACKUP", k), str)

    @staticmethod
    def _validate_environment(env: t.Mapping) -> t.NoReturn:
        """Check if provided environment conforms to the specification"""
        validate = partial(validate_helper, msg="Environment")
        for var in env.values():
            # platform
            validate(var, "model", str)
            validate(var, "cores", t.List)
            validate(var, "frequencies", t.List)
            validate(var, "schedule_tag", str)

            # paths
            validate(var, "path_apps", str)
            validate(var, "path_data", str)

            # driver settings
            validate(var, "driver_type", str)
            validate(var, "driver_params", t.Mapping)

            if var["driver_type"] in ["SSHUnixDriver", "ADBAndroidDriver"]:
                # connection
                validate(var, ("driver_params", "ip"), str)
                validate(var, ("driver_params", "port"), int)
                validate(var, ("driver_params", "user"), str)
                validate(var, ("driver_params", "group"), str)
                validate(var, ("driver_params", "key"), str)
                validate(var, ("driver_params", "gateway"), type(None), str)

    def _get_environments(self, _ignore: t.Optional[str] = None):
        """Read environments from exot's root directory"""
        environments = AttributeDict()
        env_dir = self.root_path / "environments"
        files = [f for f in env_dir.glob("*.toml") if not f.stem == _ignore]
        for file in files:
            _ = toml.load(file)
            self._validate_environment(_)
            environments[file.stem] = AttributeDict.from_dict(_)
        return environments

    @property
    def environments(self) -> AttributeDict:
        """Get available environments"""
        if not hasattr(self, "_environments"):
            self._environments = AttributeDict.from_dict(self._get_environments())

        return self._environments

    @environments.setter
    def environments(self, value: t.Mapping):
        """Set available environments"""
        if not isinstance(value, t.Mapping):
            raise TypeError(
                f"only a mapping can be assigned to environments, got: {type(value)}"
            )

        for v in value.values():
            self._validate_environment(v)

        self._environments = AttributeDict.from_dict(value)

    @property
    def environments_apps_zones(self) -> t.Mapping:
        """Get a mapping with environments, apps, zones and zone configs

        TODO (REFACTOR): There might be too many methods to parse the different
        configuration dimensions (experiment config, environment files, zones, apps).

        Returns:
            t.Mapping: A mapping with the following structure:
                       - Mapping depth 1: environment name
                       - Mapping depth 2: apps in the environment
                       - Mapping depth 3: app zone, zone_config, and app_config,
                                          standalone, sched
        """
        _out = {_: {} for _ in self.config.ENVIRONMENTS}

        for env in self.config.ENVIRONMENTS:
            for app in self.config.ENVIRONMENTS[env]["APPS"]:
                _app = self.config.ENVIRONMENTS[env]["APPS"][app]
                _zone = _app.zone
                _type = _app.get("type", "local")
                _start_individually = _app.get("start_individually", False)

                if env not in self.environments:
                    raise MisconfiguredError(
                        f"The environment {env!r} for app {app!r} is not available; "
                        f"Available environments: {list(self.environments.keys())!r}"
                    )

                if _zone not in self.environments[env]:
                    raise MisconfiguredError(
                        "zone {!r} is not available in environment {!r}".format(_zone, env)
                    )

                _is_standalone = _type == "standalone"
                _app_config = {}

                if app in self.config.ENVIRONMENTS[env]:
                    _app_config = self.config.ENVIRONMENTS[env][app]
                elif not _is_standalone and app not in self.config.ENVIRONMENTS[env]:
                    those_that_have = [_ for _ in self.config.ENVIRONMENTS[env] if _ != "APPS"]
                    raise MisconfiguredError(
                        f"App {app!r} requires a corresponding configuration in ENVIRONMENTS; "
                        f"Apps that do have one: {those_that_have!r}"
                    )

                _sched = _app.get("sched", None)
                _sched = Path(_sched).name if _sched else _sched

                driver_t = self.environments[env][_zone]["driver_type"].lower()
                is_unix = "unix" in driver_t
                is_android = "android" in driver_t

                if not any([is_unix, is_android]):
                    raise RuntimeError(
                        f"Both not unix and not Android? Driver type: {driver_t}"
                    )

                _out[env].update(
                    {
                        app: {
                            "executable": _app["executable"],
                            "type": _type,
                            "zone": _zone,
                            "args": _app.get("args", []),
                            "zone_config": self.environments[env][_zone],
                            "app_config": _app_config,
                            "standalone": _is_standalone,
                            "start_individually": _start_individually,
                            "sched": _sched,
                            "is_unix": is_unix,
                            "is_android": is_android,
                        }
                    }
                )

        return AttributeDict.from_dict(_out)

    @property
    def available_environments(self) -> list:
        """Get names of available environments"""
        assert self.config, "Config must be read first"
        assert self.environments, "Environment configurations need to be read first"

        envs = list()
        for env in self.config.ENVIRONMENTS:
            if env not in self.environments:
                raise MisconfiguredError(
                    f"The environment {env} is not in defined environments: "
                    f"{list(self.environments.keys())}"
                )
            envs.append(env)

        return envs

    def environment_config_general(self, env: str) -> AttributeDict:
        """Provides an environment-specific proxy to the general experiment config

        Args:
            env (str): The environment

        Returns:
            AttributeDict: The root or environment-specific config
        """
        if env in self.config.EXPERIMENT.GENERAL.keys():
            environment_config_general = self.config.EXPERIMENT.GENERAL.copy()
            for key in environment_config_general[env]:
                environment_config_general[key] = environment_config_general[env][key]
            del environment_config_general[env]
            return environment_config_general
        else:
            return self.config.EXPERIMENT.GENERAL

    @property
    def name(self) -> t.Optional[str]:
        """Get experiment name"""
        return self.config.name if self.configured else None

    @property
    def save_path(self) -> t.Optional[Path]:
        """Get save path for experiments"""
        return Path.joinpath(self.root_path, self.config.save_path) if self.configured else None

    @property
    def log_path(self) -> t.Optional[Path]:
        """Get experiment log path"""
        if self.configured:
            if "log_path" in self.config:
                return Path(self.config.log_path)
            else:
                return Path.joinpath(self.save_path, "_logs")
        else:
            return None

    @property
    def path(self) -> t.Optional[Path]:
        """Get the save path of the particular Experiment"""
        return Path.joinpath(self.save_path, self.config.name) if self.configured else None

    @property
    def root_path(self) -> t.Optional[Path]:
        """Get the exot root path"""
        return self._root_path

    def remote_path(self, env: str, zone: str) -> Path:
        """Get a remote experiment path given an environment and a zone"""
        assert isinstance(env, str)
        assert isinstance(zone, str)

        assert env in self.config.ENVIRONMENTS, "env in config.ENVIRONMENTS"
        assert env in self.environments, "env in environments"
        assert zone in self.environments[env], "zone in env"

        return self.remote_save_path(env, zone) / self.path.relative_to(self.save_path)

    def remote_save_path(self, env: str, zone: str) -> Path:
        """Get a remote experiment path given an environment and a zone"""
        assert isinstance(env, str)
        assert isinstance(zone, str)

        assert env in self.config.ENVIRONMENTS, "env in config.ENVIRONMENTS"
        assert env in self.environments, "env in environments"
        assert zone in self.environments[env], "zone in env"

        return Path(self.environments[env][zone].path_data)

    @property
    def channel(self) -> Channel:
        """Get the configured channel"""
        return getattr(self, "_channel", None)

    @channel.setter
    def channel(self, value: Channel) -> None:
        """Set the experiment channel"""
        if not isinstance(value, Channel):
            raise TypeError("'channel' should be a Channel-type", value)
        self._channel = value

    @property
    def drivers(self) -> t.Optional[AttributeDict]:
        """Get the experiment drivers

        Returns:
            t.Optional[AttributeDict]: A mapping: env: str -> zone: str -> driver: Driver
        """
        return getattr(self, "_drivers", None)

    @drivers.setter
    def drivers(self, value: [t.Mapping[str, t.Mapping[str, Driver]]]) -> None:
        """Set the experiment drivers

        Args:
            value (t.Mapping[str, t.Mapping[str, Driver]]):
                A mapping: env: str -> zone: str -> driver: Driver

        Raises:
            ExperimentTypeError: If keys have wrong types
            ExperimentValueError: If a environment or zone is not available
            WrongDriverError: If a leaf is not a Driver instance
        """

        _paths = get_valid_access_paths(value, _leaf_only=False)
        _leaves = leaves(value)

        if not all(all(isinstance(_, str) for _ in __) for __ in _paths):
            raise ExperimentTypeError("wrong paths in mapping (must be str)", value)
        if not all(isinstance(_, Driver) for _ in _leaves):
            raise WrongDriverError("drivers must be instances of Driver", value)

        _first_level_keys = [_[0] for _ in _paths if len(_) == 1]
        _second_level_keys = [_[1] for _ in _paths if len(_) == 2]

        for env in _first_level_keys:
            if env not in self.environments:
                raise ExperimentValueError(f"supplied driver env {env} not available")
        for zone in _second_level_keys:
            if zone not in [v.zone for v in self.config.APPS.values()]:
                raise ExperimentValueError(f"supplied driver zone {zone} not in config")

        self._drivers = AttributeDict.from_dict(value)

    @property
    def phases(self) -> t.Dict:
        """Get experiment phases"""
        return getattr(self, "_phases", None)

    @phases.setter
    def phases(self, value: t.Dict) -> None:
        """Set experiment phases"""
        if not isinstance(value, t.Dict):
            raise ExperimentValueError(
                "value set to 'phases' must be a dict-like object", value
            )

        # for key in value:
        #     valid_keys = ["train", "eval", "run"]
        #     if key not in valid_keys:
        #         raise ExperimentValueError(
        #             f"values in 'phases' should be one of: {valid_keys!r}", value
        #         )

        self._phases = value

    @property
    def _update_mode(self) -> bool:
        return self.config.experiment_exists_action == "update"

    def write(self) -> None:
        """Serialise the experiment"""
        assert self.configured, "Experiment must be configured before writing"
        assert self.bootstrapped, "Experiment must be bootstrapped before writing"

        if any(isabstract(layer) for layer in self.layers):
            raise SerialisingAbstractError("cannot serialise abstract classes")

        # handle existing experiment path
        if self.path.exists():
            # experiment_exists_action exists in the config
            if hasattr(self.config, "experiment_exists_action"):
                if self.config.experiment_exists_action == "overwrite":
                    self.logger.info(
                        f"experiment path '{self.path}' already exists, will be overwritten"
                    )
                    delete(self.path)
                elif self.config.experiment_exists_action == "move":
                    self.logger.info(move_action(self.path))
                elif self.config.experiment_exists_action == "update":
                    pass
                else:
                    raise ExperimentAbortedError("experiment directory existed")
            # experiment_exists_action does not exist in the config, prompt user
            else:
                proceed = prompt(
                    f"destination path {self.path} exists, move the old one and proceed"
                )
                if proceed:
                    self.logger.info(move_action(self.path))
                else:
                    raise ExperimentAbortedError("experiment directory existed")

        if not self._update_mode:
            if self.path.exists():
                raise ExperimentRuntimeError(f"{self.path} shouldn't exist before serialising")

        self.path.mkdir(parents=True, exist_ok=self._update_mode)

        # TODO [ACCESS_PATHS]: saving these might not be eventually required
        # NOTE: generator objects cannot be pickled/serialised
        self._phases_access_paths = list(get_valid_access_paths(self.phases, _leaf_only=True))

        # fmt: off
        # Write the experiment's configuration,  metadata, and available environments
        with self.path.joinpath("_configuration.toml").open("w")  as cfgfile,\
             self.path.joinpath("_metadata.toml").open("w")       as gitfile,\
             self.path.joinpath("_environments.toml").open("w")   as envfile:
            toml.dump(self.config.as_dict(), cfgfile)
            toml.dump({"git": self._git}, gitfile)
            toml.dump(self.environments.as_dict(), envfile)
        # fmt: on

        # Serialise the experiment object
        pickle_path = self.path / "_experiment.pickle"
        self.serialise(pickle_path)

        self.logger.debug(f"serialised experiment '{self.name}' to '{pickle_path}'")

        # Serialise all 'owned' Run objects
        def call_run_write(run):
            if not isinstance(run, Run):
                raise ExperimentTypeError("a run in phases not an instance of Run", run)
            run.write()

        call_with_leaves(call_run_write, self.phases)

    @classmethod
    def read(
        cls, path: Path, new_root_path: t.Optional[Path] = None, *, diff_and_replace: bool = False,
    ) -> Experiment:
        """Read a serialised Experiment and produce an Experiment instance

        Args:
            path (Path): The path to the serialised experiment
            new_root_path (t.Optional[Path], optional): A new root path
            diff_and_replace (bool, optional): Do file diff and replace values from the instance with those from files?

        Returns:
            Experiment: A restored instance of Experiment

        Raises:
            WrongRootPathError: If saved and current root paths cannot be resolved
        """
        instance = cls.deserialise(path)
        get_root_logger().info(f"unpicked an experiment instance {instance}")

        _original_cwd = Path.cwd()
        if new_root_path and new_root_path.resolve() != _original_cwd:
            os.chdir(new_root_path)

        # Handle reading experiments from moved directories, especially those moved
        # automatically by the framework. `path.parts[-1]` is the picke file.
        #
        # Note: this may fail when there are stark differences in how experiments are placed.
        _save_path = path.parts[-3]
        _name = path.parts[-2]

        if instance.config.save_path != _save_path:
            instance.config.save_path = _save_path

        if instance.config.name != _name:
            instance.config.name = _name

        try:
            # Check if the current and the saved root path resolve to a valid root path
            validate_root_path(instance.root_path)
            if new_root_path:
                instance._root_path = new_root_path
        except WrongRootPathError:
            get_root_logger().critical(
                f"the pickled root path {cls.root_path} and current working directory "
                f"'{Path.cwd()}' does not resolve to a valid root path."
            )
            raise WrongRootPathError(Path.cwd() / instance.root_path)
        finally:
            os.chdir(_original_cwd)

        _need_bootstrap = False
        # Check if saved config differs from that in the file
        contained_config = instance.path / "_configuration.toml"
        if contained_config.exists() and diff_and_replace:
            config_from_directory = toml.load(contained_config)
            differences = dict_diff(instance.config, config_from_directory)
            if differences:
                _need_bootstrap = True
                get_root_logger().warning(
                    f"configs from pickle and directory differ at {differences!r}"
                )
                instance.config = config_from_directory

        # Check if saved environments differ from those in the file
        contained_environments = instance.path / "_environments.toml"
        if contained_environments.exists() and diff_and_replace:
            environments_from_directory = toml.load(contained_environments)
            differences = dict_diff(instance.environments, environments_from_directory)
            if differences:
                _need_bootstrap = True
                get_root_logger().warning(
                    f"environments from pickle and directory differ at {differences!r}"
                )
                instance.environments = environments_from_directory

        # Check status of the git repository
        if getattr(instance, "_git", None):
            if GitRepository.is_git_directory(instance.root_path):
                repo = GitRepository(instance.root_path)
                if repo.commit != instance._git["commit"]:
                    get_root_logger().error(
                        "git commit of unpickled experiment repo '{}' does not match "
                        "the commit in which the unpickling was performed '{}'".format(
                            instance._git["commit"][:8], repo.commit[:8]
                        )
                    )
                if repo.dirty:
                    get_root_logger().warning("unpickling in a dirty git repository")
            else:
                get_root_logger().warning(
                    f"unpickled in a directory that is not a git directory"
                )
        if _need_bootstrap:
            instance.bootstrap()


        # Populate the experiment phases and read existing Run's
        instance.phases = {tag: {} for tag in instance.config.EXPERIMENT.PHASES}
        instance.estimated_duration = {
            tag: instance.config.EXPERIMENT.GENERAL.delay_after_bootstrap
            if "delay_after_bootstrap" in instance.config.EXPERIMENT.GENERAL
            else 10.0
            for tag in instance.config.EXPERIMENT.PHASES
        }


        # Deserialise all contained Runs
        for pickled_run in instance.path.rglob("*/_run.pickle"):
            run_query = run_path_unformatter(pickled_run)
            # TODO [ACCESS_PATHS]: setitem will only allow setting existing keys, but
            # these can be checked explicitly, as long as access paths are available.
            if not (
                run_query in instance._phases_access_paths
                ):
                get_root_logger().warning(f"{run_query!r} must be a valid query")
            setitem(
                instance.phases, run_query, instance.run_type.read(pickled_run, parent=instance), force=True
            )
        instance._phases_access_paths = list(get_valid_access_paths(instance.phases, _leaf_only=True))

        return instance

        instance = integrate(instance)

    def backup(self, _with_logs: bool = False, _with_results: bool = False) -> None:
        """Archive an Experiment and, if possible, upload to a backup server

        Args:
            _with_logs (bool, optional): Backup contained experiment logs?
            _with_results (bool, optional): Backup contained experiments results?

        Raises:
            InvalidBackupPathError: If backup path is invalid
        """
        assert self.configured, "experiment must be configured before backing up"
        assert self.path.exists(), "experiment must be serialised before backing up"

        if getitem(self.config, "backup_path", None):
            where = self.root_path / Path(self.config.backup_path)
        else:
            where = self.save_path / "_backup"

        if not where.is_dir():
            try:
                where.mkdir(parents=True)
            except OSError as e:
                _ = InvalidBackupPathError(f"backup path '{where}' not valid")
                _.args += (e, e.args)
                raise _

        file = Path("{}.tbz".format(self.config.name))
        file = add_timestamp(file)
        path = Path.joinpath(where, file)

        self.logger.info(f"archiving experiment '{self.name}' in '{path}'")

        files_to_backup = (
            f
            for f in self.path.glob("**/*.*")
            if not (
                (_with_logs and "_logs" in f.parent.parts)
                or (_with_results and "_results" in f.parent.parts)
            )
        )

        with tarfile.open(path, "x:bz2") as _:
            for file in files_to_backup:
                _.add(file)

        # If BACKUP section is available in the config, attempt to send the archive
        if self.configured and "BACKUP" in self.config:
            self.logger.info(f"sending experiment '{self.name}' archive to remote backup")

            _ = file_backup(path, self.config.BACKUP.as_dict())
            if not _.ok:
                self.logger.error(
                    f"failed to send experiment archive, exited: {_.exited}, "
                    f"stderr: {_.stderr}"
                )

    @abc.abstractmethod
    def generate(self) -> None:
        """Populate the experiment phases and instantiate Run's

        Preconditions:
            - The experiment should be configured and bootstrapped.
        """
        pass


    """
    The following methods are used to estimate the duration of the experiment.
    """

    @property
    def estimated_duration(self) -> t.Dict:
        """Get durations of experiment phases"""
        return getattr(self, "_estimated_duration", None)

    @estimated_duration.setter
    def estimated_duration(self, value: t.Dict) -> None:
        """Set experiment phases"""
        if not isinstance(value, t.Dict):
            raise ExperimentValueError(
                "value set to 'duration' must be a dict-like object", value
            )

        self._estimated_duration = value

    def print_duration(self) -> None:
        """Prints the estimated experiment duration
        """
        assert self.estimated_duration, "Experiment must be generated first"
        total_duration = 0

        for phase in self._estimated_duration:
            time = datetime.timedelta(seconds=self.estimated_duration[phase])
            self.logger.info(
                f"Estimated duration of a single repetition of the {phase} phase is {str(time)}"
            )
            total_duration += (
                self._estimated_duration[phase]
                * self.config["EXPERIMENT"]["PHASES"][phase]["repetitions"]
            )

        time = datetime.timedelta(seconds=(total_duration))
        self.logger.info(
            f"This results in a total estimated duration of {str(time)} for all repetitions."
        )

    def map_to_runs(
        self, function: t.Callable[[t.Any], t.Any], *, parallel: bool = True
    ) -> t.Generator:
        """Map a function to the Runs concurrently

        Args:
            function (t.Callable[[t.Any], t.Any]): A callable with sig. (Any) -> Any
            parallel (bool, optional): run callable concurrently?

        Returns:
            t.Generator: The executor map generator
        """
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(
                thread_name_prefix="MapperThread"
            ) as executor:
                return executor.map(function, leaves(self.phases))
        else:
            return map(function, leaves(self.phases))

    def _create_drivers(self) -> t.Mapping:
        """Create experiment drivers

        Returns:
            t.Mapping: A mapping with environment as the 1st level, zone as the 2nd,
                       leaves are Driver objects created with a DriverFactory

        Raises:
            DriverCreationError: If driver could not be instantiated
        """
        assert self.configured, "must be configured before adding drivers"
        assert self.environments, "must have environments before adding drivers"

        driver_factory = DriverFactory()
        drivers = {k: {} for k in self.config.ENVIRONMENTS}
        for env in drivers:
            for app in self.config.ENVIRONMENTS[env].APPS:
                try:
                    _zone = self.config.ENVIRONMENTS[env].APPS[app].zone
                    _zone_params = self.environments[env][_zone]
                    _driver_type = _zone_params["driver_type"]
                    _driver_params = _zone_params["driver_params"]
                    drivers[env][_zone] = driver_factory(_driver_type, backend=_driver_params)
                except (KeyError, AttributeError) as e:
                    raise DriverCreationError(e)

        return drivers

    def _validate_execute(self, env: str) -> t.NoReturn:
        """Verify that the experiment can be executed

        Also creates drivers, if they're not available.

        Raises:
            ExperimentRuntimeError: If an experiment Run is not digested
            MisconfiguredError: If the execution cannot be performed due to config
        """
        assert self.configured, "must be configured before execution"
        assert self.bootstrapped, "must be bootstrapped before execution"
        assert self.environments, "must have environments before execution"

        # check for missing environments
        _ = [env_k for env_k in self.config.ENVIRONMENTS if env_k not in self.environments]
        if _:
            msg = "environments {m!r} in config not in available ({a!r})".format(
                m=_, a=list(self.environments.keys())
            )
            self.logger.critical(msg)
            raise MisconfiguredError(msg)

        # check for missing zones
        for env_k in self.config.ENVIRONMENTS:
            for app_k, app_v in self.config.ENVIRONMENTS[env_k].APPS.items():
                if app_v.zone not in self.environments[env_k]:
                    msg = "zone {z!r} of app {a!r} not available for env {e!r}".format(
                        z=app_v.zone, a=app_k, e=env_v
                    )
                    self.logger.critical(msg)
                    raise MisconfiguredError(msg)

        # check if runs are digested
        if any(not run.digested for run in leaves(self.phases)):
            msg = "some experiment runs were not digested"
            self.logger.critical(msg)
            raise ExperimentRuntimeError(msg)

        # check if drivers are created
        if self.drivers:
            for driver in leaves(self.drivers):
                try:
                    driver.disconnect()
                except (AssertionError, RuntimeError):
                    pass

        self.drivers = self._create_drivers()

        # Connection check only necessary for actual environment
        if not all(_.can_connect() for _ in leaves(self.drivers[env])):
            msg = "at least one driver cannot connect"
            self.logger.critical(msg)
            raise MisconfiguredError(msg)

    @property
    def execution_status(self) -> dict:
        """Gets the experiment execution status

        The execution status has the following form:

            {
                <phase_key>: {
                    <run_key>: [
                        (<env name 1>, [True, True, ..., False]),
                        (<env name 2>, [True, True, ..., False]),
                        ...
                    ],
                    ...
                },
                ...
            }

        Returns:
            dict: A dict with a structure like `phases`, with lists as leaves, containing
                tuples (env, [executed runs]).
        """
        return map_to_leaves(
            lambda run: [
                (env, list(v.values()))
                #for env, v in run.infer_execution_status(update=update_runs).items()
                for env, v in run.execution_status.items()
            ],
            self.phases,
            _seq=False,
        )

        #if not hasattr(self, "_execution_status"):
        #    # get from runs if not yet available

        #return self._execution_status


    #def _get_execution_status(
    #    self, env: t.Optional[str] = None, query: t.Optional[t.Union[str, tuple]] = None
    #) -> dict:
    #    """Queries the execution status with optional filtering with env and query parameters

    #    Args:
    #        env (t.Optional[str], optional): The environment
    #        query (t.Optional[t.Union[str, tuple]], optional): The query to `getitem`

    #    Returns:
    #        dict: The exection status, possibly filtered
    #    """
    #    if not env:
    #        return self.execution_status
    #    else:
    #        if query:
    #            query_result = getitem(self.execution_status, query)
    #            index_of_env = query_result.index(next(_ for _ in query_result if _[0] == env))
    #            return getitem(self.execution_status, query + (index_of_env,))[1]
    #        else:
    #            return map_to_leaves(
    #                lambda x: list(filter(lambda v: v[0] == env, x))[0][1],
    #                self.execution_status,
    #                _seq=False,
    #            )

    def infer_execution_status(self) -> dict:
        """Infers the execution status from contained Runs, optionally update self and Runs

        Args:

        Returns:

        Raises:
            TypeError: Wrong types provided to keyword arguments
        """
        for phase in self.phases.values():
            for run in phase.values():
                run.infer_execution_status().items()

    def _modify_execution_status(
        self, env: str, query: t.Union[str, tuple], value: bool, *, push: bool = False
    ) -> None:
        """Modifies the execution status of experiment's phases and runs

        Args:
            env (str): The environment
            query (t.Union[str, tuple]): The query to `getitem`
            value (bool): The value to set
            push (bool, optional): Push to runs?
        """
        query_result = getitem(self.execution_status, query)
        index_of_env = query_result.index(next(_ for _ in query_result if _[0] == env))

        setitem(self.execution_status, query + (index_of_env,), (env, value))

        if push:
            self._push_execution_status()

    def _push_execution_status(self) -> None:
        """Pushes execution status to experiment's runs
        """
        for stats, run in zip(leaves(self.execution_status), leaves(self.phases)):
            for env, rep_stats in stats:
                run._execution_status[env] = {i: v for i, v in enumerate(rep_stats)}

    def execute_in_environment(
        self, env: str, phases: t.List[str], resume: bool = False
    ) -> None:
        """Executes the experiment in a specific environment

        Args:
            env (str): The environment
            phases (t.List[str]): The phases to execute
            resume (bool, optional): Resume execution?

        Returns:
            None: If no phases are executed

        Raises:
            ExperimentAbortedError: Interrupted by the user or options are mismatched
            ExperimentRuntimeError: Failed to send or timed out
            TypeError: Wrong type supplied to 'phases'
            ValueError: Any of the provided phases were not available
        """
        if resume and not getattr(self, "_update_mode", False):
            self.logger.critical(
                "'resume' should only be used if the 'experiment_exists_action' is 'update'"
            )
            proceed = prompt(
                f"Do you want to continue resuming execution in env: {env!r} "
                "with experiment action 'overwrite'"
            )

            if not proceed:
                raise ExperimentAbortedError("Aborted due to 'resume' settings.")

        if not isinstance(phases, t.List):
            raise TypeError(f"'phases' must be a list, got: {type(phases)}")

        invalid_phases = [_ for _ in phases if _ not in self.phases]
        if invalid_phases:
            raise ValueError(f"some/all of provided phases not available: {invalid_phases}")

        self._validate_execute(env)

        phases_to_execute = {
            phase: values for phase, values in self.phases.items() if phase in phases
        }

        if not phases_to_execute:
            self.logger.warning(f"no phases specified ({phases!r}), exiting")
            return None

        runs = (
            # If resuming, pick runs that have at least one rep that has not been executed...
            (
                run
                for run in leaves(phases_to_execute)
                if not all(list(run.execution_status[env].values()))
            )
            if resume
            # Otherwise, pick all runs.
            else leaves(phases_to_execute)
        )

        bootstrapping_driver = None
        _general = self.environment_config_general(env)

        try:
            for run in runs:
                self.logger.info(f"executing phases: {phases!r} in env {env!r}")

                if True:
                #try:
                    run.send(env)
                    run.env_path(env).mkdir(parents=True, exist_ok=True)
                    run.drivers_proxy = self.drivers[env]

                    for zone, driver in self.drivers[env].items():
                        driver.mkdir(str(run.remote_env_path(env, zone)), parents=True)

                        if not bootstrapping_driver:
                            bootstrapping_driver = (zone, driver)
                            self.logger.info(f"{env}->{zone}: configuring to {_general}")
                            driver.setstate(**_general)
                            _ = driver.getstate()
                            self.logger.debug(f"{env}->{zone}: current state: {_}")

                    run.execute(env, resume=resume)
                    run.fetch_and_cleanup(env)
                #except (KeyboardInterrupt, SystemExit) as e:
                #    raise
                #except Exception as e:
                #    self.logger.critical(f"{env}: exception: {e} in run {run}")
        except (KeyboardInterrupt, SystemExit) as e:
            self.logger.critical(f"{env}: execution interrupted: {type(e)}")
            raise
        finally:
            for zone, driver in self.drivers[env].items():
                if not driver.backend.connected:
                    driver.backend.connect()

                if (zone, driver) == bootstrapping_driver:
                    self.logger.info(f"{env}->{zone}: cleaning up")
                    # clean up explicitly before disconnecting to log/inspect state
                    driver.cleanup()
                    self.logger.debug(f"{env}->{zone}: cur. state: {driver.getstate()!r}")

                delete_ok = driver.delete(path=str(self.remote_path(env, zone)), recursive=True)
                self.logger.info(
                    f"{env}->{zone}: {'successfully' if delete_ok else 'failed to'} "
                    "deleted remote data directory"
                )

                driver.disconnect()
                self.logger.info(f"{env}->{zone}: disconnected")

    def execute(
        self, *, env_phase_mapping: t.Optional[t.Dict[str, t.List[str]]] = None, resume=False
    ) -> None:
        """Execute the Experiment on a target platform

        Preconditions:
            - The experiment should be configured and bootstrapped.
            - Drivers should be available and should be able to connect.
            - Each run should be digested.
        """

        if env_phase_mapping:
            if not isinstance(env_phase_mapping, t.Dict):
                raise TypeError("env_phase_mapping must be a dict")

            for env, phases in env_phase_mapping.items():
                if not isinstance(env, str) or env not in self.config.ENVIRONMENTS:
                    raise ValueError(f"invalid env in env_phase_mapping: {env}")

                if not isinstance(phases, t.List):
                    raise TypeError(f"invalid value type in env_phase_mapping: {type(phases)}")

                invalid_phases = [_ for _ in phases if _ not in self.phases]
                if invalid_phases:
                    raise ValueError(f"invalid phases for env {env!r}: {invalid_phases}")

        for env in self.config.ENVIRONMENTS:
            self._validate_execute(env)
            try:
                phases = env_phase_mapping[env] if env_phase_mapping else list(self.phases)
                self.execute_in_environment(env, phases=phases, resume=resume)
            except (KeyboardInterrupt, SystemExit, Exception) as e:
                exc = traceback.format_exception(type(e), e, e.__traceback__)
                self.logger.critical(
                    f"execution failed for environment {env} with exception: {exc}"
                )
                raise ExperimentExecutionFailed(exc)


"""
Run
----------

Base class for experiment runs.
"""


class Run(
    Pickleable,
    Configurable,
    HasParent,
    IntermediatesHandler,
    parent=Experiment,
    metaclass=abc.ABCMeta,
):
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialise a Run

        Args:
            *args (t.Any): Passed to parent initialisers
            **kwargs (t.Any): Passed to parent initialisers

        Raises:
            ExperimentRuntimeError: If not configured or doesn't have a parent
        """
        Configurable.__init__(self, *args, **kwargs)
        HasParent.__init__(self, *args, **kwargs)

        if not self.configured or not self.parent:
            raise ExperimentRuntimeError(
                "Run instances are expected to be instantiated completely "
                "with config and parent"
            )

        self.logger = self.parent.logger.getChild("Run")
        self._runtime_config = dict()

        if "repetitions" not in self.config:
            raise ExperimentRuntimeError(
                "All run instances and subclasses require a 'repetitions' config."
            )

        self._execution_status = {
            env: {rep: False for rep in range(self.config.repetitions)}
            for env in self.parent.config.ENVIRONMENTS
        }

    def _length_helper(self, length: t.Optional[int]) -> int:
        """Check if length type/value or get configured length

        Args:
            length (t.Optional[int]): The length
        """
        pass

    def _configure_layers_proxy(self, which: str, **kwargs) -> None:
        """Configure runtime-configurable layers

        Layers are configured with own config (phase, length_seconds, frequency), and
        optionally with values in kwargs. The layer keys must match (e.g. 'io', 'src')!

        Since the config contains only the frequency, the bit rate is added too.

        Args:
            which (str): "encode" or "decode"
            **kwargs: keyword arguments to pass to the parent Experiment configurator,
            keys should correspond to layer names (e.g. 'lne', 'io')
        """
        pass

    @property
    def execution_status(self) -> dict:
        """Gets the execution status of the Run

        The Run's execution status has the form:

            {
                <env name>: {
                    <rep idx>: True | False,
                    ...
                },
                ...
            }

        Returns:
            dict: The execution status
        """
        return getattr(self, "_execution_status", self.infer_execution_status())

    def infer_execution_status(self, *, update: bool = True) -> dict:
        """Infers the execution status from the contained files

        Returns:
            dict: The inferred execution status

        Args:
            update (bool, optional): Update the Run's own execution status?

        Raises:
            TypeError: Wrong type provided to 'update'
        """
        if not isinstance(update, bool):
            raise TypeError(f"'update' must be a {bool}, got: {type(update)}")

        envs = list(self.parent.config.ENVIRONMENTS)
        status = dict.fromkeys(envs)

        for env in envs:
            logs = self.env_path(env).glob("*/*.log.csv")
            reps = sorted(set([log_path_unformatter(log)[1] for log in logs]))
            total_reps = list(range(self.config.repetitions))
            status[env] = {rep: (rep in reps) for rep in total_reps}

        if update:
            self._execution_status = status

        return status

    @property
    def runtime_config(self) -> dict:
        """Get the last used runtime_config."""
        return getattr(self, "_runtime_config", dict())

    @runtime_config.setter
    def runtime_config(self, value) -> None:
        """Set the rt config"""
        if not isinstance(value, dict):
            raise TypeError("'value' should be a dict-type", value)
        setattr(self, "_runtime_config", cp.deepcopy(value))

    def __repr__(self) -> str:
        _ = ""
        _ += "ingested, " if self.ingested else ""
        _ += "digested, " if self.digested else ""
        return f"<Run at {hex(id(self))} ({_}config={self.config})>"

    @classmethod
    def read(cls, path: Path, parent: t.Optional[object] = None) -> object:
        """Read a serialised Run and produce a Run instance

        Args:
            path (Path): The path of the serialised instance
            parent (t.Optional[object], optional): The parent experiment to set

        Returns:
            object: The deserialised instance
        """
        instance = cls.deserialise(path)
        get_root_logger().debug(f"unpicked a run instance {instance}")
        if parent:
            instance.parent = parent
        return instance

    def _write_custom_schedules(self) -> t.List[Path]:
        """Write all custom schedules to the run path

        Returns:
            t.List[Path]: A list with copied schedules

        Raises:
            MisconfiguredError: If a provided file does not exist
        """
        apps_with_schedules = {}
        copied_schedules = []

        for app in [v["APPS"] for v in self.parent.config.ENVIRONMENTS.values()]:
            for k, v in {k: v for k, v in app.items() if "sched" in v}.items():
                apps_with_schedules.update({k: v})

        for app_name, app in apps_with_schedules.items():
            sched_path = Path(app["sched"])

            if not sched_path.exists():
                raise MisconfiguredError(
                    f"The custom schedule file {sched_path} for app {app_name!r} does not exist!"
                )

            new_path = self.path / sched_path.name
            copied_schedules.append(copy(sched_path, new_path, replace=True))

        return copied_schedules

    def write(self) -> None:
        """Serialise a Run instance"""
        file_to_write = "_run.pickle"
        path = Path.joinpath(self.path, file_to_write)
        self.path.mkdir(parents=True, exist_ok=True)
        self.serialise(path)
        self.logger.debug(f"serialised run to '{path}'")

        with (self.path / "_config.toml").open("w") as _:
            toml.dump(self.config, _)

        written_schedules = self._write_custom_schedules()
        self.logger.debug(f"wrote custom schedules: {[str(_) for _ in written_schedules]}")

    @property
    @abc.abstractmethod
    def identifier(self) -> str:
        """Get the save path"""
        pass

    @property
    def path(self) -> Path:
        formatted_directory = run_path_formatter(self.config.phase, self.identifier)
        return Path.joinpath(self.parent.path, formatted_directory)

    def env_path(self, env: str, *, relative: bool = False) -> Path:
        """Get a specific environment's path"""
        assert isinstance(env, str)
        assert env in self.parent.config.ENVIRONMENTS, "env in config.ENVIRONMENTS"
        assert env in self.parent.environments, "env in environments"

        if relative:
            return (self.path / env).relative_to(self.parent.save_path)
        else:
            return self.path / env

    def rep_path(self, env: str, rep: int, *, relative: bool = False) -> Path:
        assert isinstance(rep, int), f"Wrong type for rep, should be int but is {type(rep)}"
        assert rep < self.config.repetitions, f"rep out of range, has to be small than {self.config.repetitions}"
        return self.env_path(env, relative=relative).joinpath(repetition_formatter(rep))

    def remote_path(self, env: str, zone: str) -> Path:
        """Get a remote path given an environment and a zone"""
        assert isinstance(env, str)
        assert isinstance(zone, str)

        assert env in self.parent.config.ENVIRONMENTS, "env in config.ENVIRONMENTS"
        assert env in self.parent.environments, "env in environments"
        assert zone in self.parent.environments[env], "zone in env"

        return self.parent.environments[env][zone].path_data / self.path.relative_to(
            self.parent.save_path
        )

    def remote_env_path(self, env: str, zone: str) -> Path:
        """Get a remote environment path given an environment and an zone"""
        return self.remote_path(env, zone) / env

    def remote_rep_path(self, env: str, zone: str, rep : int) -> Path:
        return self.remote_env_path(env, zone) / repetition_formatter(rep)

    @property
    def drivers_proxy(self) -> t.Mapping[str, Driver]:
        """Get a proxy to the Experiment's drivers for a specific environment

        Returns:
            t.Mapping[str, Driver]: A mapping: zone: str -> driver: Driver
        """
        return getattr(self, "_drivers_proxy", {})

    @drivers_proxy.setter
    def drivers_proxy(self, value: t.Mapping[str, Driver]) -> None:
        """Set a proxy to the Experiment's drivers for a specific environment

        Args:
            value (t.Mapping[str, Driver]): A mapping: zone: str -> driver: Driver
        """
        assert isinstance(value, t.Mapping), "must be a Mapping"
        assert all(isinstance(_, Driver) for _ in value.values()), "must be Drivers"
        self._drivers_proxy = value

    @property
    def ingested(self) -> bool:
        """Has the run performed all decoding steps?"""
        return all(
            (_ is not None and len(_ if _ is not None else [])) for _ in self.i_streams.values()
        )

    def clear_ingest(self) -> None:
        """Has the run performed all decoding steps?"""
        for stream in self.i_streams:
            setattr(self, 'i_' + stream, None)

    @property
    def path_ingestion_data(self) -> Path:
        _filepath = self.path
        for elem in self.ingestion_tag:
            _filepath = _filepath / elem
        return _filepath

    def load_ingestion_data(self, prefix: t.Optional[str] = '', bundled: bool = False, **kwargs) -> None:
        self.clear_ingest()
        self._configure_layers_proxy("decode", **kwargs)
        self.update_ingestion_tag()
        self.intermediates = AttributeDict()
        if bundled:
            i_streams = self.load_mapping_bundled(path=self.path_ingestion_data, prefix=prefix+"stream.i_")
        else:
            i_streams = self.load_mapping(path=self.path_ingestion_data, prefix=prefix+"stream.i_")
        for stream in i_streams:
            setattr(self, "i_" + stream, i_streams[stream])
        self.intermediates = self.load_mapping(path=self.path_ingestion_data, prefix=prefix+"im_")
        if not self.ingested:
            raise Exception("Loading ingestion data failed due to missing data!")

    def save_ingestion_data(self, prefix: t.Optional[str] = '', bundled: bool = False) -> None:
        if self.ingested:
            if bundled:
                self.save_mapping_bundled("i_streams",     path=self.path_ingestion_data, prefix=prefix+"stream.i_")
                self.save_mapping_bundled("intermediates", path=self.path_ingestion_data, prefix=prefix+"im_")
            else:
                self.save_mapping("i_streams",     path=self.path_ingestion_data, prefix=prefix+"stream.i_")
                self.save_mapping("intermediates", path=self.path_ingestion_data, prefix=prefix+"im_")
        else:
            get_root_logger().warning("Run not ingested, nothing to be saved!")

    def remove_ingestion_data(self, prefix: t.Optional[str] = '') -> None:
        self.clear_ingest()
        self.intermediates = AttributeDict()
        self.remove_mapping("i_streams",     path=self.path_ingestion_data, prefix=prefix+"stream.i_")
        self.remove_mapping("intermediates", path=self.path_ingestion_data, prefix=prefix+"im_")

    def update_ingestion_tag(self) -> None:
        self.ingestion_tag = (self.parent.layers.io.config.env, repetition_formatter(self.parent.layers.io.config.rep))

    @property
    def ingestion_tag(self) -> bool:
        return getattr(self, '_ingestion_tag', ('', ''))

    @ingestion_tag.setter
    def ingestion_tag(self, value: tuple) -> None:
        if isinstance(value, tuple):
            self._ingestion_tag = value
        else:
            raise TypeError(f"Ingestion Tag has to be of type tuple, but is {type(value)}")

    @property
    def digested(self) -> bool:
        """Has the run performed all encoding steps?"""
        return all(
            (_ is not None and len(_ if _ is not None else [])) for _ in self.o_streams.values()
        )

    @abc.abstractmethod
    def digest(self, **kwargs) -> None:
        """Perform all encoding steps"""
        pass

    @abc.abstractmethod
    def ingest(self, **kwargs) -> None:
        """Perform all decoding steps"""
        pass

    def make_random_bitarray(self, length: t.Optional[int] = None) -> bitarray:
        """Generate a random bit array of specified or configured length"""
        return bitarray(self.make_random_boolarray(length).tolist())

    def make_random_boolarray(self, length: t.Optional[int] = None) -> np.ndarray:
        """Generate a random bool NumPy array of specified or configured length"""
        return np.random.randint(0, 2, self._length_helper(length), dtype=np.dtype("bool"))

    def make_random_intarray(self, length: t.Optional[int] = None) -> np.ndarray:
        """Generate a random bool NumPy array of specified or configured length"""
        return np.random.randint(0, 2, self._length_helper(length), dtype=np.dtype("int"))

    def make_alternating_boolarray(self, length: t.Optional[int] = None) -> np.ndarray:
        """Generate an alternating NumPy bool array of specified or configured length"""
        return np.resize([True, False], self._length_helper(length))

    def make_alternating_intarray(self, length: t.Optional[int] = None) -> np.ndarray:
        """Generate an alternating binary NumPy int array of specified or configured length"""
        return np.resize([1, 0], self._length_helper(length))

    def make_alternating_bitarray(self, length: t.Optional[int] = None) -> bitarray:
        """Generate an alternating bit array of specified or configured length"""
        return bitarray(self.make_alternating_boolarray(length).tolist())

    def make_constant_boolarray(
        self, value: bool, length: t.Optional[int] = None
    ) -> np.ndarray:
        """Generate an constant NumPy bool array of specified or configured length"""
        assert isinstance(value, bool), "Value must be a boolean"
        return np.resize([value], self._length_helper(length))

    def make_constant_intarray(self, value: int, length: t.Optional[int] = None) -> np.ndarray:
        """Generate an constant binary NumPy int array of specified or configured length"""
        assert isinstance(value, int) and value in [0, 1], "Value must be either 1 or 0"
        return np.resize([value], self._length_helper(length))

    def make_constant_bitarray(self, value: bool, length: t.Optional[int] = None) -> bitarray:
        """Generate an constant bit array of specified or configured length"""
        return bitarray(self.make_constant_boolarray(value, length).tolist())

    def make_repeated_intarray(
        self, base: np.ndarray, length: t.Optional[int] = None
    ) -> np.ndarray:
        """Generate a NumPy int array of specified or configured length with a repeated pattern

        Args:
            base (np.ndarray): The base for the repeated pattern
            length (t.Optional[int], optional): The length

        Returns:
            np.ndarray: The generated array
        """
        return np.resize(base, self._length_helper(length))

    def _bootstrap_apps(self, env: str, rep: int) -> t.Mapping:
        """Bootstrap apps for the experiment

        Args:
            env (str): The environment name in which the app is to be executed
            rep (int): The repetition number

        Returns:
            t.Mapping: A mapping with apps as keys, and values of:
                       - executable
                       - zone
                       - config
                       - runtime arguments

        Raises:
            ExperimentRuntimeError: If an application was not executable
        """
        apps = self.parent.environments_apps_zones[env].copy()

        for app in apps:
            self.logger.debug(f"bootstrapping app {app!r} in env: {env!r}, run: {self}")
            _e_z = "{}->{}".format(env, apps[app].zone)
            _driver = self.drivers_proxy[apps[app].zone]

            # Common values

            sched = (
                apps[app]["sched"]
                if apps[app]["sched"] and app != "src"
                else apps[app]["zone_config"]["schedule_tag"] + ".sched"
            )

            apps[app].sched = sched
            apps[app].process = None
            apps[app].duration = self.estimated_duration()

            if apps[app].is_android:
                apps[app].intent = None

            # Standalone apps

            if apps[app].standalone:
                if not _driver.executable_exists(apps[app].executable):
                    raise ExperimentRuntimeError(
                        "{}: app {!r} standalone executable {!r} not valid".format(
                            _e_z, app, apps[app].executable
                        )
                    )

                if apps[app].is_unix:
                    # Save stdout and stderr
                    apps[app].args += [
                        "1>{}".format(
                            str(
                                self.remote_rep_path(env, apps[app].zone, rep)
                                / generic_log_formatter(app, dbg=False)
                            )
                        ),
                        "2>{}".format(
                            str(
                                self.remote_rep_path(env, apps[app].zone, rep)
                                / generic_log_formatter(app, dbg=True)
                            )
                        ),
                    ]

                if apps[app].is_android and apps[app].app_config:
                    self.logger.debug(
                        "standalone android app has config, transforming into an Intent"
                    )

                    apps[app].intent = Intent(**apps[app].app_config.as_dict())
                    apps[app].intent.es.update({'filename':str(self.remote_path(env, apps[app].zone) / apps[app].sched)})

                cfg = apps[app].app_config.as_dict()
                filename = app_configfile_formatter(app)
                with self.rep_path(env, rep).joinpath(filename).open("w") as file:
                    json.dump(cfg, file, indent=2)

                self.logger.debug(f"bootstrapped standalone app {app!r}")
                continue

            # Framework apps

            # On Unix, prepend the path to executables.
            if apps[app].is_unix:
                apps[app].executable = str(
                    Path(apps[app]["zone_config"]["path_apps"]) / apps[app].executable
                )

            if not _driver.executable_exists(apps[app].executable):
                raise ExperimentRuntimeError(
                    "{}: app {!r} executable {!r} not available or executable on target".format(
                        _e_z, app, apps[app].executable
                    )
                )

            # These keys are needed by default. Will be added if not already preset.
            apps[app].app_config["meter"] = apps[app].app_config.get("meter", dict())
            apps[app].app_config["logging"] = apps[app].app_config.get("logging", dict())
            apps[app].app_config["schedule_reader"] = apps[app].app_config.get(
                "schedule_reader", dict()
            )
            apps[app].app_config["meter"]["period"] = self.parent.environment_config_general(
                env
            ).sampling_period
            apps[app].app_config["logging"]["app_log_filename"] = str(
                self.remote_rep_path(env, apps[app].zone, rep)
                / app_log_formatter(app, dbg=False)
            )
            apps[app].app_config["logging"]["debug_log_filename"] = str(
                self.remote_rep_path(env, apps[app].zone, rep)
                / app_log_formatter(app, dbg=True)
            )

            if sched:
                apps[app].app_config["schedule_reader"]["input_file"] = str(
                    self.remote_path(env, apps[app].zone) / sched
                )
                apps[app].app_config["schedule_reader"]["reading_from_file"] = True
            else:
                apps[app].app_config["schedule_reader"]["reading_from_file"] = False
                self.logger.warning(f"App {app!r} does not have a schedule file?")

            cfg = apps[app].app_config.as_dict()
            filename = app_configfile_formatter(app)
            with self.rep_path(env, rep).joinpath(filename).open("w") as file:
                json.dump(cfg, file, indent=2)

            apps[app].json = json.dumps(cfg)
            apps[app].args = ["--json_string", json.dumps(cfg)]

            self.logger.debug(f"bootstrapped local app {app!r}")

        return apps

    def execute(self, env: str, resume=False) -> None:
        """Execute a single Run in an environment

        Args:
            env (str): An environment name
        """
        active_wait = (
            True
            if "active_wait" not in self.parent.environment_config_general(env)
            else self.parent.environment_config_general(env).active_wait
        )

        # If resuming, pick repetitions that have not been executed
        reps = [
            rep
            for rep, executed in self.execution_status[env].items()
            if ((not executed) if resume else True)
        ]

        self.logger.info(
            f"executing in env: {env!r}, resume: {resume}, reps: {reps!r}, run: {self!r}"
        )

        for rep in reps:
            self.rep_path(env, rep).mkdir(parents=True, exist_ok=True)
            for zone, driver in self.drivers_proxy.items():
                driver.mkdir(str(self.remote_rep_path(env, zone, rep)), parents=True)
            self.logger.info(
                f"executing in env: {env}, rep: {rep}, estimated duration: "
                f"{self.estimated_duration()}s, run: {self!r}"
            )

            apps = self._bootstrap_apps(env, rep)
            if not all(
                [
                    "src" in apps,
                    "snk" in apps if "Exploratory" not in type(self).__name__ else True,
                ]
            ):
                raise ExperimentRuntimeError("apps must at least have a 'src' and 'snk'")

            master_processes = {zone: [] for zone in self.drivers_proxy}
            slave_processes = {zone: [] for zone in self.drivers_proxy}
            auxiliary_processes = {zone: [] for zone in self.drivers_proxy}
            standalone_processes = {zone: [] for zone in self.drivers_proxy}

            start_threads = []
            barrier = threading.Barrier(1 + len(self.drivers_proxy))

            try:
                # 0. Init state
                if "src" in apps:
                    self.drivers_proxy[apps["src"].zone].initstate()
                else:
                    self.drivers_proxy[list(self.drivers_proxy)[0]].initstate()

                # 1. Launch non-src apps with start_individually != True
                for app in [
                    _
                    for _ in apps
                    if _ != "src" and not apps[_].get("start_individually", False)
                ]:
                    apps[app].process, _ = self.drivers_proxy[apps[app].zone].spawn(
                        apps[app].executable, apps[app].args, details=apps[app]
                    )
                    slave_processes[apps[app].zone].append(apps[app].process)
                    # TODO specific to persistent
                    if apps[app].process.exited is not None:
                        self.logger.error(
                            f"{app!r} executable exited prematurely ({apps[app].process.exited}), stderr: {_.stderr}"
                        )
                        raise ExperimentRuntimeError(f"{app!r} executable exited prematurely")
                    else:
                        self.logger.debug(
                            f"spawned a {app!r} executable: {apps[app].process!r}"
                        )

                # 2. Launch the src app and add other apps as slaves
                apps["src"].process, _ = self.drivers_proxy[apps["src"].zone].spawn(
                    apps["src"].executable,
                    apps["src"].args,
                    slaves=slave_processes[apps["src"].zone],
                    details=apps["src"],
                )
                master_processes[apps["src"].zone].append(apps["src"].process)
                if apps["src"].process.exited is not None:
                    self.logger.error(
                        f"'src' executable exited prematurely ({apps[app].process.exited}), stderr: {_.stderr}"
                    )
                    raise ExperimentRuntimeError(f"'src' executable exited prematurely")
                else:
                    self.logger.debug(f"spawned a 'src' executable: {apps['src'].process!r}")

                # 3. Delay between spawn & start
                sleep_amount = (
                    1.0
                    if "delay_after_spawn" not in self.parent.environment_config_general(env)
                    else self.parent.environment_config_general(env).delay_after_spawn
                )
                self.logger.debug(f"sleeping after spawning regular apps for {sleep_amount}s")
                sleep(sleep_amount)

                # 4. Start auxiliary apps with start_individually == True
                for app in [_ for _ in apps if apps[_].get("start_individually", False)]:
                    apps[app].process, _ = self.drivers_proxy[apps[app].zone].spawn(
                        apps[app].executable, apps[app].args, details=apps[app]
                    )
                    auxiliary_processes[apps[app].zone].append(apps[app].process)

                    if apps[app].process.exited is not None:
                        self.logger.error(
                            f"auxiliary executable {app!r} exited prematurely ({apps[app].process.exited}), "
                            f"stderr: {_.stderr}"
                        )
                        raise ExperimentRuntimeError(f"{app!r} executable exited prematurely")
                    else:
                        self.logger.debug(
                            f"spawned an auxiliary executable {app!r}, id: {apps[app].process!r}"
                        )

                sleep_amount = (
                    1.0
                    if "delay_after_auxiliary"
                    not in self.parent.environment_config_general(env)
                    else self.parent.environment_config_general(env).delay_after_auxiliary
                )
                self.logger.debug(f"sleeping after spawning auxiliary apps for {sleep_amount}s")
                sleep(sleep_amount)

                # 5. Start apps
                start_pgroup = {zone: [] for zone in self.drivers_proxy}
                start_before_pgroup = {zone: [] for zone in self.drivers_proxy}

                for zone in start_pgroup:
                    start_pgroup[zone] += slave_processes[zone] + master_processes[zone]
                    # filder processes that are not meant to be started with the starter below
                    start_pgroup[zone] = [
                        _ for _ in start_pgroup[zone] if _ not in standalone_processes[zone]
                    ]

                    start_before_pgroup[zone] = [
                        _ for _ in slave_processes[zone] if _ not in standalone_processes[zone]
                    ]

                    # There can be apps that do not have src/snk apps
                    #if not start_pgroup[zone]:
                    #    self.logger.critical(f"no processes to start in zone {zone!r}")
                    #    raise ExperimentRuntimeError(f"no processes to start in zone {zone!r}")

                stagger = (
                    self.parent.environment_config_general(env)
                    if "stagger" in self.parent.environment_config_general(env)
                    else True
                )

                # Should start staggered?
                if stagger:

                    def starter(_zone, _driver, _start_pgroup, _start_before_pgroup, _barrier):
                        self.logger.debug(f"waiting on barrier for zone {_zone}")
                        _barrier.wait()
                        if _start_before_pgroup:
                            _driver.start(*_start_before_pgroup)
                        if _start_pgroup:
                            _driver.start(*_start_pgroup)

                    for zone, driver in self.drivers_proxy.items():
                        self.logger.debug(
                            f"will start staggered in zone {zone!r}, "
                            f"first processes: {start_before_pgroup[zone]!r}, "
                            f"then processes: {master_processes[zone]!r}"
                        )

                        start_threads.append(
                            threading.Thread(
                                target=starter,
                                args=(
                                    zone,
                                    driver,
                                    master_processes[zone],
                                    start_before_pgroup[zone],
                                    barrier,
                                ),
                            )
                        )
                else:

                    def starter(_zone, _driver, _processes, _barrier):
                        self.logger.debug(f"waiting on barrier for zone {_zone}")
                        _barrier.wait()
                        if _processes:
                            _driver.start(*_processes)

                    for zone, driver in self.drivers_proxy.items():
                        self.logger.debug(
                            f"will start simultaneously in zone {zone!r}, processes: {start_pgroup[zone]!r}"
                        )

                        start_threads.append(
                            threading.Thread(
                                target=starter, args=(zone, driver, start_pgroup[zone], barrier)
                            )
                        )

                for thread in start_threads:
                    thread.start()

                self.logger.debug(f"waiting on barrier for starting threads")
                barrier.wait()
                self.logger.debug(f"passed barrier for starting threads")

                for thread in start_threads:
                    thread.join()

                for zone in start_pgroup:
                    self.logger.debug(
                        f"started processes: {start_pgroup[zone]} in zone: {zone}"
                    )

                if not active_wait:
                    for zone, driver in self.drivers_proxy.items():
                        self.logger.debug(f"disconnecting driver for zone {zone!r}")
                        driver.backend.disconnect()

                # 6. Wait for SRC to complete
                timeout_duration = (
                    10 * self.estimated_duration()
                    if "timeout_duration" not in self.parent.environment_config_general(env)
                    else self.parent.environment_config_general(env).timeout_duration
                )

                timeout_duration = max(timeout_duration, 3.0)
                sleep_wait_for_src = max(1.5 * self.estimated_duration(), 1.0)

                with Timeout(timeout_duration, throwing=False) as timeout:
                    # Inactive wait:
                    if not active_wait:
                        total_waited = 0.0

                        while True:
                            sleep(sleep_wait_for_src)
                            for zone, driver in self.drivers_proxy.items():
                                if not driver.backend.connected:
                                    self.logger.debug(f"reconnecting driver for zone {zone!r}")
                                    driver.backend.connect()
                            if apps["src"].process.exited is not None:
                                break
                            else:
                                total_waited += sleep_wait_for_src
                                self.logger.debug(
                                    f"src app did not exit after {total_waited} sleep time "
                                    f"in zone {zone!r}"
                                )
                                _ = 0.5 * sleep_wait_for_src
                                __ = 0.1 * self.estimated_duration()
                                sleep_wait_for_src = max(max(_, __), __)
                    else:
                        # Active wait:
                        self.drivers_proxy[apps["src"].zone].wait(
                            apps["src"].process,
                            refresh_period=max(0.275 * self.estimated_duration(), 1.0),
                        )

                # 7. Check exit status of 'src' app, stop & kill.
                if timeout.timed_out or apps["src"].process.exited is None:
                    self.logger.error(
                        f"'src' executable timed out after {timeout_duration} seconds, "
                        "will be terminated"
                        if timeout.timed_out
                        else f"'src' executable failed or timed out"
                    )

                # Stop, then kill the 'src' app.
                self.drivers_proxy[apps["src"].zone].stop(apps["src"].process)
                sleep(0.25)
                self.drivers_proxy[apps["src"].zone].kill(apps["src"].process)

                if apps["src"].process.exited != 0:
                    self.logger.error(
                        f"'src' executable exited with non-zero return code "
                        f"({apps['src'].process.exited!r})"
                    )
                else:
                    self.logger.debug("'src' executable exited successfully!")

                # 8. Check exit statuses and explicitly kill apps
                for app in [_ for _ in apps if _ != "src"]:
                    self.drivers_proxy[apps[app].zone].stop(apps[app].process)
                    sleep(0.25)

                    if apps[app].process.exited != 0:
                        self.logger.error(
                            f"app {app!r} executable exited with non-zero return code "
                            f"({apps[app].process.exited!r})"
                        )
                    else:
                        self.logger.debug(f"{app!r} executable exited successfully!")
                        break

                    if apps[app].process.exited is None:
                        self.logger.warning(f"app {app!r} did not exit gracefully!")

                # 9. Update execution state for rep to True if no exception was fired
                self._execution_status[env][rep] = True

            except (KeyboardInterrupt, SystemExit):
                self.logger.critical(f"interrupted repetition {rep} in env: {env!r}")
                raise
            except ExperimentRuntimeError as e:
                self.logger.error(f"runtime error in repetition {rep} in env: {env!r}: {e}")
                continue
            finally:
                # 10. Kill all processes
                self.logger.debug("shutting down any remaining apps")
                for zone, driver in self.drivers_proxy.items():
                    all_processes = [
                        *master_processes[zone],
                        *slave_processes[zone],
                        *auxiliary_processes[zone],
                    ]
                    self.logger.debug(f"killing all_processes: {all_processes!r}")
                    driver.kill(*all_processes)

    def send(self, env):
        previous_id = None
        previous_wd = None
        for zone, driver in self.parent.drivers[env].items():
            # if drivers connect to zones that have the same IP/serial address, the next driver
            # must force connection, because the first driver to connect will lock the platform...
            current_id = driver.connection_id
            if not driver.connected:
                driver.connect(force=current_id == previous_id)  # gets original settings
            current_wd = driver.working_directory

            self.parent.logger.info(f"{env}->{zone}: connected: {driver.__class__.__name__}")
            self.parent.logger.debug(f"{env}->{zone}: original state: {driver.original_state!r}")
            self.parent.logger.debug(f"{env}->{zone}: curr_id: {current_id}, prev_id: {previous_id}")
            self.parent.logger.debug(f"{env}->{zone}: curr_wd: {current_wd}, prev_wd: {previous_wd}")

            if current_id == previous_id and current_wd == previous_wd:
                self.parent.logger.debug(
                    f"{env}->{zone}: experiment already sent, ids and working directories were "
                    f"the same in the previous driver ({previous_id}->{previous_wd!r})"
                )
            else:
                self.parent.logger.info(f"{env}->{zone}: sending experiment")
                if driver.exists(str(self.parent.remote_path(env, zone))):
                    self.parent.logger.debug(
                        f"{env}->{zone}: remote experiment directory exists, will be deleted"
                    )
                    _ = driver.delete(str(self.parent.remote_path(env, zone)), recursive=True)
                    if _:
                        self.parent.logger.debug(
                            f"{env}->{zone}: deleted successfully: {self.parent.remote_path(env, zone)!s}"
                        )
                    else:
                        self.parent.logger.debug(
                            f"{env}->{zone}: failed to delete: {self.parent.remote_path(env, zone)!s}"
                        )

                with Timeout(60, throwing=False) as timeout:
                    _ = driver.send(path_from=self.path, path_to=Path.joinpath(self.parent.remote_path(env, zone), self.path.relative_to(self.parent.path)))
                    if not _.ok:
                        _msg = f"{env}->{zone}: failed to send: {_.stderr}"
                        self.parent.logger.critical(_msg)
                        raise ExperimentRuntimeError(_msg)
                    else:
                        self.parent.logger.info(f"{env}->{zone}: experiment sent!")

                if timeout.timed_out:
                    driver.disconnect()

                    raise ExperimentRuntimeError(
                        f"{env}->{zone}: Timed out after 30s during experiment sending"
                    )

            previous_id = cp.copy(current_id)
            previous_wd = cp.copy(current_wd)

    def fetch_and_cleanup(self, env):
        previous_id = None
        previous_wd = None

        for zone, driver in self.parent.drivers[env].items():
            current_id = driver.connection_id
            current_wd = driver.working_directory

            if current_id == previous_id and current_wd == previous_wd:
                self.parent.logger.debug(
                    f"{env}->{zone}: skipping fetching due to same id and working directory "
                    f"in the previous driver ({previous_id}->{previous_wd})"
                )
            else:
                fetch_result = driver.fetch(
                    path_from=Path.joinpath(self.parent.remote_path(env, zone), self.path.relative_to(self.parent.path)), path_to=self.path
                )
                if not fetch_result.ok:
                    _msg = f"{env}->{zone}: failed to fetch logs: {fetch_result.stderr}"
                    self.parent.logger.critical(_msg)
                else:
                    self.parent.logger.info(f"{env}->{zone}: experiment logs fetched!")

                driver.delete(path=str(Path.joinpath(self.parent.remote_path(env, zone), self.path.relative_to(self.parent.path))), recursive=True)

            previous_id = cp.copy(current_id)
            previous_wd = cp.copy(current_wd)

    def estimated_duration(self, env=None) -> t.Optional[float]:
        """Get the estimated duration of this Run's execution

        Returns:
            t.Optional[float]: the duration in seconds, or None if not digested
        """
        raise Exception(NotImplemented)

