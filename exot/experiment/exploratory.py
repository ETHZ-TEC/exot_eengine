"""Capacity bound experiment"""
import typing as t
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from exot.exceptions import *
from exot.util.attributedict import AttributeDict
from exot.util.misc import safe_eval, validate_helper

from ._base import Experiment, Run
from ._mixins import *

__all__ = "ExploratoryExperiment"


class ExploratoryExperiment(Experiment, type=Experiment.Type.Exploratory):
    """Stub class for a capacity experiment
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = ExploratoryRun

    @staticmethod
    def required_layers():
        return ["io"]

    def validate(self):
        """Validate experiment configuration"""

        # In addition to the validation performed in the parent class...
        super().validate()

        _ = partial(validate_helper, self.config, msg="Exploratory")

        # ...verify experiment phases configuration
        for k in self.config.EXPERIMENT.PHASES:
            _(("EXPERIMENT", "PHASES", k), AttributeDict)
            _(("EXPERIMENT", "PHASES", k, "rdpstreams"), t.List)
            _(("EXPERIMENT", "PHASES", k, "repetitions"), int)

        # ... verify general zone/platform settings
        _(("EXPERIMENT", "GENERAL"), AttributeDict)
        _(("EXPERIMENT", "GENERAL", "latency"), int)
        _(("EXPERIMENT", "GENERAL", "fan"), bool, str, list)
        _(("EXPERIMENT", "GENERAL", "governors"), str, list)
        _(("EXPERIMENT", "GENERAL", "frequencies"), str, float, list)
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
        self.estimated_duration = {tag: 0.0 for tag in self.config.EXPERIMENT.PHASES}

        for phase, values in self.config.EXPERIMENT.PHASES.items():
            rdpstreams = values["rdpstreams"]

            if isinstance(rdpstreams, str):
                self.logger.info(
                    f"rdpstreams in phase {phase!r} given as a str, will be evaluated"
                )
                rdpstreams = safe_eval(rdpstreams)

            if not isinstance(rdpstreams, (t.List, np.ndarray)):
                raise GenerateTypeAssertion("rdpstreams must be a list")

            types = set(type(x) for x in rdpstreams)
            if not all(issubclass(_, pd.core.frame.DataFrame) for _ in types):
                raise GenerateTypeAssertion(
                    f"rdpstreams should only be pandas DataFrames, but were: {types}"
                )

            rdpstream_id = 0
            for rdpstream in rdpstreams:
                self.logger.debug(f"generating run for phase: {phase}, rdpstream: {rdpstream}")
                self.phases[phase][rdpstream_id] = ExploratoryRun(
                    config=AttributeDict(
                        phase=phase,
                        rdpstream_id=rdpstream_id,
                        rdpstream=rdpstream,
                        repetitions=values["repetitions"],
                    ),
                    parent=self,
                )

                # Perform all encodings
                self.phases[phase][rdpstream_id].digest()
                self.estimated_duration[phase] += (
                    self.phases[phase][rdpstream_id].estimated_duration()
                    + self.estimated_delays_duration
                )
                rdpstream_id += 1
            del self.config.EXPERIMENT.PHASES[phase]["rdpstreams"]


"""
ExploratoryRun
--------------

Exploratory runs are considered to be immutable during the Experiment execution. Once
the configuration is provided, the instance is 'frozen' and the config should not be
updated.

Values that are always provided to layers at runtime:
- From a Run's config: phase, bit_count, trace, repetitions,
- Obtained from a Run's config and the parent: bit_rate.

Values in the parent Experiment can be easily accessed through the parent proxy:
`self.parent`.
"""


class ExploratoryRun(
    Run,
    StreamHandler,
    Irdpstream, Irawstream,
    Ordpstream, Orawstream, Oschedules,
    serialise_save=["o_rdpstream", "o_rawstream", "o_schedules", "i_rawstream", "i_rdpstream"],
    parent=ExploratoryExperiment,
):
    @property
    def identifier(self):
        return self.config.rdpstream_id

    @classmethod
    def read(cls, path: Path, parent: t.Optional[object] = None) -> object:
        instance = super().read(path, parent)
        instance.load_data_bundled()
        return instance

    def write(self) -> None:
        """Serialises the ExploratoryRun

        In addition to the base class'es `write`, the ExploratoryRun als0 writes an
        archive with output streams and writes the schedules to *.sched files.
        """
        super().write()
        self.save_data_bundled()
        self.write_schedules()

    @property
    def required_config_keys(self):
        return ["phase", "rdpstream", "repetitions"]

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

        _ = {layer: {} for layer in layers}
        for layer in _:
            _[layer].update(self.config)
            _[layer].update(
                bit_rate=None,
                subfrequency=None,
                environments_apps_zones=self.parent.environments_apps_zones,
                path=self.path,
            )

            if which == "decode" and self.digested:
                _[layer].update(**self.o_streams)

            if layer in kwargs:
                _[layer].update(**kwargs.pop(layer))

            self.logger.debug(f"configuring {which} of layer {layer}")

        configurator(**_)

    def digest(self, *, skip_checks: bool = False, write_intermediates: bool = False, **kwargs) -> None:
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

        self.logger.debug("<--- digesting begun! --->")
        self.logger.debug("producing rawstream <- rdpstream")
        self.o_rdpstream = self.config.rdpstream
        del self.config.rdpstream  # TODO not sure if that is a good idea...
        self.o_rawstream = self.parent.layers.io.encode(self.o_rdpstream, skip_checks)
        self.logger.debug("setting schedules   <- individual schedules from the i/o layer")
        self.o_schedules = self.parent.layers.io.schedules
        self.collect_intermediates(self.parent.layers)
        self.logger.debug("<--- digesting completed! --->")

    def write_schedules(self) -> t.List[Path]:
        """Write experiment schedules to files

        Returns:
            t.List[Path]: a list of paths where schedules were written
        """
        assert self.parent, "must have a parent experiment"
        assert self.o_schedules, "must have output schedules"
        self._configure_layers_proxy("encode")
        return self.parent.layers.io.write_schedules(self.o_schedules)

    def ingest(self, *, skip_checks: bool = False, **kwargs) -> None:
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

        if "skip_io_decode" not in kwargs:
            self.logger.debug(
                "producing rdpstream <- choosing data and preprocessing rawstream"
            )
            self.i_rdpstream = self.parent.layers.io.decode(self.i_rawstream, skip_checks)
        else:
            self.logger.debug("skipped producing rdpstream")

        self.collect_intermediates(self.parent.layers)

    def estimated_duration(self, env=None) -> t.Optional[float]:
        return self.o_rdpstream.timestamp.sum()

