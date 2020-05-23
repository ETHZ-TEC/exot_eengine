"""Capacity bound experiment"""
import typing as t
from functools import partial
from pathlib import Path

import copy as cp
from shutil import copyfile

import numpy as np
import pandas as pd

from exot.exceptions import *
from exot.util.attributedict import AttributeDict
from exot.util.misc import safe_eval, validate_helper
from exot.util.wrangle import run_path_formatter

from ._base import Experiment, Run
from ._mixins import *

__all__ = "AppExecExperiment"


class AppExecExperiment(Experiment, type=Experiment.Type.AppExec):
    """Stub class for an experiment where a src app executes a given schedule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = AppExecRun

    @staticmethod
    def required_layers():
        return ["io"]

    def validate(self):
        """Validate experiment configuration"""

        # In addition to the validation performed in the parent class...
        super().validate()

        _ = partial(validate_helper, self.config, msg="AppExec")

        # ...verify experiment phases configuration
        for k in self.config.EXPERIMENT.PHASES:
            _(("EXPERIMENT", "PHASES", k), AttributeDict)
            _(("EXPERIMENT", "PHASES", k, "repetitions"), int)

        # ... verify general zone/platform settings
        _(("EXPERIMENT", "GENERAL"), AttributeDict)
        _(("EXPERIMENT", "GENERAL", "governors"), str, list, type(None))
        _(("EXPERIMENT", "GENERAL", "frequencies"), str, float, list, type(None))
        _(("EXPERIMENT", "GENERAL", "sampling_period"), float)

    def _length_helper(self, length: t.Optional[int]) -> int:
        """Check if length type/value or get configured length

        Args:
            length (t.Optional[int]): The length
        """
        if length:
            assert isinstance(length, int), "Length must be an integer"
            assert length > 0, "Length must be greater than zero"
            return length
        else:
            return 0

    def generate(self):
        assert self.configured, "Experiment must be configured before generating"
        assert self.bootstrapped, "Experiment must be bootstrapped before generating"
        self.phases = {tag: {} for tag in self.config.EXPERIMENT.PHASES}
        cp_phases   = cp.deepcopy(self.config.EXPERIMENT.PHASES)

        self.estimated_duration = {
            tag: self.config.EXPERIMENT.GENERAL.delay_after_bootstrap
            if "delay_after_bootstrap" in self.config.EXPERIMENT.GENERAL
            else 10.0
            for tag in self.config.EXPERIMENT.PHASES
        }
        for phase, values in cp_phases.items():
            repetitions = values.pop("repetitions")
            for runname, runparam in values.items():
                if "schedules" not in runparam:
                    raise LookupError("No schedules specified!")
                schedules = runparam['schedules']
                if not isinstance(schedules, (list)):
                    raise GenerateTypeAssertion("schedules must be a list")
                types = set(type(x) for x in schedules)
                if not all(issubclass(_, str) for _ in types):
                    raise GenerateTypeAssertion(
                        f"Schedules should only be strings, but were: {types}"
                    )

                if "environments" not in runparam:
                    raise LookupError("No environments specified!")
                envs = runparam['environments']
                if not isinstance(envs, (list)):
                    raise GenerateTypeAssertion("environments must be a list")
                types = set(type(x) for x in envs)
                if not all(issubclass(_, str) for _ in types):
                    raise GenerateTypeAssertion(
                        f"Envirnoments should only be strings, but were: {types}"
                    )

                if "durations" not in runparam:
                    durations = [None for _ in range(len(envs))]
                else:
                    durations = runparam['durations']

                o_schedules = {}
                for env, sched, dur in zip(envs, schedules, durations):
                    o_schedules[env] = (sched, dur)

                self.logger.debug(f"generating run for phase: {phase}, run name {runname}")
                self.phases[phase][runname] = AppExecRun(
                    config=AttributeDict(
                        phase=phase,
                        o_schedules=o_schedules,
                        name=runname,
                        repetitions=repetitions,
                    ),
                    parent=self,
                )
                # Perform all encodings
                self.phases[phase][runname].digest()
                self.estimated_duration[phase] += (
                    self.phases[phase][runname].estimated_duration()
                    + self.estimated_delays_duration
                )

"""
AppExecRun
--------------

AppExec runs are considered to be immutable during the Experiment execution. Once
the configuration is provided, the instance is 'frozen' and the config should not be
updated.

Values that are always provided to layers at runtime:
- From a Run's config: phase, bit_count, trace, repetitions,
- Obtained from a Run's config and the parent: bit_rate.

Values in the parent Experiment can be easily accessed through the parent proxy:
`self.parent`.
"""


class AppExecRun(
    Run,
    StreamHandler,
    Ibitstream, Isymstream, Ilnestream, Irdpstream, Irawstream,
    serialise_save=[],
    serialise_ignore=["i_rawstream", "i_rdpstream", "i_lnestream", "i_symstream", "i_bitstream"],
    parent=AppExecExperiment,
):
    @property
    def path(self):
        formatted_directory = run_path_formatter(self.config.phase, self.config.name)
        return Path.joinpath(self.parent.path, formatted_directory)

    @classmethod
    def read(cls, path: Path, parent: t.Optional[object] = None) -> object:
        instance = super().read(path, parent)
        instance.load_data()
        return instance

    def write(self) -> None:
        """Serialises the AppExecRun

        In addition to the base class'es `write`, the AppExecRun als0 writes an
        archive with output streams and writes the schedules to *.sched files.
        """
        super().write()
        self.save_data()
        self.write_schedules()

    @property
    def required_config_keys(self):
        return ["phase", "o_schedules", "repetitions", "name"]

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
        assert which in ["encode", "decode"], "'which' must be 'encode' or 'decode'"
        self.runtime_config[which] = kwargs.copy()
        layers = (
            self.parent.layers_with_runtime_encoding
            if which == "encode"
            else self.parent.layers_with_runtime_decoding
        )
        configurator = (
            self.parent.configure_layers_encoding
            if which == "encode"
            else self.parent.configure_layers_decoding
        )

        if not layers:
            return

        if 'env' in kwargs:
            config = {layer: {'env':kwargs['env']} for layer in layers}
        else:
            config = {layer: {} for layer in layers}
        for layer in config:
            config[layer].update(self.config)
            config[layer].update(
                environments_apps_zones=self.parent.environments_apps_zones,
                path=self.path,
            )

            if which == "decode" and self.digested:
                config[layer].update(**self.o_streams)

            if layer in kwargs:
                config[layer].update(**kwargs.pop(layer))

            self.logger.debug(f"configuring {which} of layer {layer}")

        configurator(**config)

    def digest(self, *, skip_checks: bool = False, **kwargs) -> None:
        """Perform all encoding operations, propagating the streams to subsequent layers

        Caveats:
            Since `digest` accesses parent config in `_configure_layers_proxy`, the digest
            step cannot be performed concurrently.

        Args:
            **kwargs: Optional configuration for layers, can be supplied in the form of:
                      layer=dict(a=..., b=...), or **{layer: {"a":...}}
        """
        assert self.parent, "must have a parent experiment"
        self._configure_layers_proxy("encode", **kwargs)

        self.collect_intermediates(self.parent.layers)
        self.logger.debug("<--- digesting completed! --->")

    def write_schedules(self) -> t.List[Path]:
        """Write experiment schedules to files

        Returns:
            t.List[Path]: a list of paths where schedules were written
        """
        assert self.parent, "must have a parent experiment"
        assert self.config.o_schedules, "must have output schedules"
        for env in self.config.o_schedules:
          copyfile(self.config.o_schedules[env][0], Path.joinpath(self.path, env + '.sched'))

    def ingest(self, *, skip_checks: bool = True, **kwargs) -> None:
        """Perform all decoding operations, propagating the streams to preceding Layers

        Args:
            **kwargs: Optional configuration for layers, can be supplied in the form of:
                      layer=dict(a=..., b=...), or **{layer: {"a":...}}
        """
        assert self.parent, "must have a parent experiment"
        self._configure_layers_proxy("decode", **kwargs)
        self.update_ingestion_tag()

        self.logger.debug("<--- ingesting begun! --->")
        self.logger.debug("producing rawstream <- reading raw measurements")
        self.i_rawstream = self.parent.layers.io.get_measurements()
        self.logger.debug("producing rdpstream <- choosing data and preprocessing rawstream")
        self.i_rdpstream = self.parent.layers.io.decode(self.i_rawstream, skip_checks)
        if 'rdp' in self.parent.layers and self.parent.layers.rdp.configured:
            self.logger.debug("producing lnestream <- preprocessing rdpstream")
            self.i_lnestream = self.parent.layers.rdp.decode(self.i_rdpstream, skip_checks)
            if 'lne' in self.parent.layers and self.parent.layers.lne.configured:
                self.logger.debug("producing symstream <- decoding lnestream")
                self.i_symstream = self.parent.layers.lne.decode(self.i_lnestream, skip_checks)
                if 'src' in self.parent.layers and self.parent.layers.src.configured:
                    self.logger.debug("producing bitstream <- decoding symstream")
                    self.i_bitstream = self.parent.layers.src.decode(self.i_symstream, skip_checks)
                else:
                    self.i_bitstream = np.full((1,1),np.nan)
            else:
                self.i_symstream = np.full((1,1),np.nan)
                self.i_bitstream = np.full((1,1),np.nan)
        else:
            self.i_lnestream = np.full((1,1),np.nan)
            self.i_symstream = np.full((1,1),np.nan)
            self.i_bitstream = np.full((1,1),np.nan)
        self.logger.debug("<--- ingesting completed! --->")
        self.collect_intermediates(self.parent.layers)

    def estimated_duration(self, env=None) -> t.Optional[float]:
        """Get the estimated duration of this Run's execution

        Returns:
            t.Optional[float]: the duration in seconds, or None if not digested
        """
        if self.digested:
            if env is not None:
                return self.config.o_schedules[env][1]
            else:
                duration = 0
                for loop_env in self.config.o_schedules:
                    duration = max([duration, self.config.o_schedules[loop_env][1]]) + 2
                return duration
        else:
            return None
