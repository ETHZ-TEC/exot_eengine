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
"""Performance evaluation experiment"""

import copy
import typing as t
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from exot.exceptions import *
from exot.util.attributedict import AttributeDict
from exot.util.misc import safe_eval, validate_helper
from exot.util.scinum import count_errors_robust
from exot.util.wrangle import Matcher, run_path_formatter

from ._base import Experiment, Run
from ._mixins import *

__all__ = ("PerformanceExperiment", "PerformanceRun")


class PerformanceExperiment(
    Experiment, serialise_save=["performance_metrics"], type=Experiment.Type.Performance
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = PerformanceRun

    @staticmethod
    def required_layers():
        return ["io", "lne", "rdp", "src"]

    def validate(self):
        """Validate experiment configuration"""

        # In addition to the validation performed in the parent class...
        super().validate()

        _ = partial(validate_helper, self.config, msg="PerformanceExperiment")

        # ...verify experiment phases configuration
        for k in self.config.EXPERIMENT.PHASES:
            _(("EXPERIMENT", "PHASES", k), AttributeDict)
            _(("EXPERIMENT", "PHASES", k, "bit_count"), int)
            _(("EXPERIMENT", "PHASES", k, "symbol_rates"), str, list)
            _(("EXPERIMENT", "PHASES", k, "repetitions"), int)

        # ... verify general zone/platform settings
        _(("EXPERIMENT", "GENERAL", "latency"), int)
        _(("EXPERIMENT", "GENERAL", "fan"), bool, str, list)
        _(("EXPERIMENT", "GENERAL", "governors"), str, list)
        _(("EXPERIMENT", "GENERAL", "frequencies"), str, float, list)
        _(("EXPERIMENT", "GENERAL", "sampling_period"), float)

    def generate(self, **kwargs):
        assert self.configured, "Experiment must be configured before generating"
        assert self.bootstrapped, "Experiment must be bootstrapped before generating"

        self.phases = {tag: {} for tag in self.config.EXPERIMENT.PHASES}
        self.estimated_duration = {
            tag: self.config.EXPERIMENT.GENERAL.delay_after_bootstrap
            if "delay_after_bootstrap" in self.config.EXPERIMENT.GENERAL
            else 10.0
            for tag in self.config.EXPERIMENT.PHASES
        }

        for phase, values in self.config.EXPERIMENT.PHASES.items():
            symbol_rates = values["symbol_rates"]

            if isinstance(symbol_rates, str):
                self.logger.info(
                    f"symbol rates in phase {phase!r} given as a str, will be evaluated"
                )
                symbol_rates = safe_eval(symbol_rates)

            if not isinstance(symbol_rates, (t.List, np.ndarray)):
                raise GenerateTypeAssertion("symbol rates must be a list")

            types = set(type(x) for x in symbol_rates)
            if not all(issubclass(_, (float, np.float, int, np.integer)) for _ in types):
                raise GenerateTypeAssertion(
                    f"symbol rates should only be int's, but were: {types}"
                )

            symbol_rate_id = 0
            for symbol_rate in symbol_rates:
                self.logger.debug(
                    f"generating run for phase: {phase}, symbol rate: {symbol_rate}"
                )
                self.phases[phase][symbol_rate_id] = PerformanceRun(
                    config=AttributeDict(
                        phase=phase,
                        bit_count=values["bit_count"],
                        symbol_rate=symbol_rate,
                        symbol_rate_id=symbol_rate_id,
                        repetitions=values["repetitions"],
                    ),
                    parent=self,
                )

                # Perform all encodings
                self.phases[phase][symbol_rate_id].digest(**kwargs)
                self.estimated_duration[phase] += (
                    self.phases[phase][symbol_rate_id].estimated_duration()
                    + self.estimated_delays_duration
                )
                symbol_rate_id += 1

    def write(self):
        super().write()
        self.save_data_bundled(prefix="data")

    @classmethod
    def read(cls, *args, **kwargs) -> object:
        instance = super().read(*args, **kwargs)
        instance.load_data_bundled(prefix="data")

        return instance

    def analyse_run_pair(
        self,
        train_phase: str,
        eval_phase: t.Optional[str],
        idx: t.Hashable,
        ingest_args: dict,
        *,
        standalone: bool = True,
        **kwargs,
    ) -> t.Tuple[pd.DataFrame, Run, t.Optional[Run]]:
        if not isinstance(train_phase, str):
            raise TypeError(f"'train_phase' must be a {str}, got: {type(train_phase)}")
        if not isinstance(eval_phase, (str, type(None))):
            raise TypeError(
                f"'eval_phase' must be a {(str, type(None))}, got: {type(eval_phase)}"
            )

        matcher = kwargs.pop("matcher", ingest_args.get("io", {}).get("matcher", None))
        if not isinstance(matcher, Matcher):
            raise TypeError(f"'matcher' should be of type {Matcher}, got: {type(matcher)}")

        description = "{}_{}".format(matcher._quantity, matcher._method)
        ingest_args["io"] = ingest_args.get("io", {})
        ingest_args["io"].update(matcher=matcher)

        if any(_ not in ingest_args["io"] for _ in ["rep", "env", "matcher"]):
            raise ValueError("some of the required keys missing from I/O layer config")

        def calculate_error_rate(ground_truth, real_sequence):
            gt = (
                np.array(ground_truth)
                if not isinstance(ground_truth, np.ndarray)
                else ground_truth
            )
            rs = (
                np.array(real_sequence)
                if not isinstance(real_sequence, np.ndarray)
                else real_sequence
            )
            return float(count_errors_robust(gt, rs)) / gt.size

        def get_run_analysis(run, train_phase=None):
            assert run.ingested, "run must be ingested"

            return {
                "phase": run.config.phase,
                "trained_with": train_phase if train_phase else "",
                "variable": description,
                "matcher": repr(matcher),
                "environment": self.layers.io.config.env,
                "repetition": self.layers.io.config.rep,
                "bit_rate": self.layers.src.symrate_to_bitrate(run.config.symbol_rate),
                "symbol_rate": run.config.symbol_rate,
                "bit_error": calculate_error_rate(run.i_bitstream, run.o_bitstream),
                "symbol_error": calculate_error_rate(run.i_symstream, run.o_symstream),
            }

        analysis_data_holder = []

        train_run = self.phases[train_phase][idx]
        eval_run = None

        if not train_run.digested:
            raise ValueError(f"run {train_run!r} was not digested!")

        train_ingest_args = copy.deepcopy(ingest_args)
        train_run.ingest(**train_ingest_args)
        analysis_data_holder.append(get_run_analysis(train_run))
        decision_device = copy.deepcopy(self.layers.lne.decision_device)

        if eval_phase:
            eval_run = self.phases[eval_phase][idx]
            if eval_run.config.symbol_rate != train_run.config.symbol_rate:
                raise RuntimeError(
                    f"eval phase symbol rate != train phase symbol rate for "
                    f"symbol rate id: {idx}, train: {train_phase}, "
                    f"eval: {eval_phase}"
                )
            if not eval_run.digested:
                raise ValueError(f"eval run {eval_run!r} was not digested!")
            if train_ingest_args["io"]["rep"] not in range(eval_run.config.repetitions):
                raise ValueError(
                    f'train repetition {train_ingest_args["io"]["rep"]} '
                    f"not available in eval phase {eval_phase}"
                )

            eval_run._configure_layers_proxy("decode")
            requested_env = train_ingest_args["io"]["env"]
            available_envs = self.layers.io.get_available_environments()

            if requested_env not in available_envs:
                msg = (
                    f"train run env {requested_env} not in "
                    f"eval run's available envs: {available_envs}"
                )
                self.logger.error(msg)
                raise RuntimeError(msg)

            eval_ingest_args = train_ingest_args.copy()
            eval_ingest_args["lne"].update(decision_device=decision_device)
            eval_run.ingest(**eval_ingest_args)
            analysis_data_holder.append(get_run_analysis(eval_run, train_phase))

        if standalone:
            return pd.DataFrame(analysis_data_holder), train_run, eval_run
        else:
            return analysis_data_holder

    def calculate_performance_metrics(
        self,
        *,
        phase_mapping: t.Dict[str, str] = {"train": "eval"},
        envs: t.List[str] = [],
        reps: t.List[int] = [],
        **kwargs,
    ) -> pd.DataFrame:
        """Calculates performance metrics for different environments, repetitions, and train->eval mappings

        Args:
            phase_mapping (t.Dict[str, str], optional): Phase mapping train -> eval
            envs (t.List[str], optional): List of environments
            reps (t.List[int], optional): List of repetitions
            **kwargs: Keyword arguments to ingest and (optionally) the matcher
        """

        if not isinstance(phase_mapping, t.Dict):
            raise TypeError(
                f"'phase_mapping' argument must be a dict, got: {type(phase_mapping)}"
            )
        if not isinstance(envs, t.List):
            raise TypeError(f"'envs' argument must be a list, got: {type(envs)}")
        if not isinstance(reps, t.List):
            raise TypeError(f"'reps' argument must be a list, got: {type(reps)}")
        if not all([isinstance(_, int) for _ in reps]):
            raise ValueError(f"'reps' argument must contain only integers")

        all_phases = list(phase_mapping.keys()) + list(phase_mapping.values())
        if not all([isinstance(_, str) for _ in phase_mapping]):
            raise ValueError("train phase mappings cannot be None")
        if not all([isinstance(_, (str, type(None))) for _ in all_phases]):
            raise ValueError(
                f"'phase_mapping' argument must contain only str->[str|None] mappings"
            )

        invalid_phases = [_ for _ in all_phases if _ is not None and _ not in self.phases]
        if invalid_phases:
            raise ValueError(f"some/all of provided phases not available: {invalid_phases}")

        invalid_envs = [_ for _ in envs if _ not in self.config.ENVIRONMENTS]
        if invalid_envs:
            raise ValueError(f"some/all of provided envs not available: {invalid_envs}")

        if not envs:
            self.logger.warning("no envs provided, will analyse all available envs")

        if "matcher" not in kwargs and "matcher" not in kwargs.get("io", {}):
            raise ValueError(
                "a 'matcher' should be provided as a keyword argument or "
                "a keyword argument to the I/O layer"
            )

        matcher = kwargs.pop("matcher", kwargs.get("io", {}).get("matcher", None))
        if not isinstance(matcher, Matcher):
            raise TypeError(f"'matcher' should be of type {Matcher}, got: {type(matcher)}")

        description = "{}_{}".format(matcher._quantity, matcher._method)
        ingest_args = kwargs.copy()
        ingest_args["io"] = ingest_args.get("io", {})
        ingest_args["io"].update(matcher=matcher)

        analysis_data_holder = []

        for train_phase, eval_phase in phase_mapping.items():
            self.logger.info(f"analysing performance for phases: {train_phase} -> {eval_phase}")

            for idx in self.phases[train_phase]:
                train_run = self.phases[train_phase][idx]

                all_reps = np.arange(train_run.config.repetitions).tolist()
                reps = reps if reps else all_reps
                if any([rep for rep in reps if rep not in all_reps]):
                    raise ValueError(
                        "provided reps ({}) invalid for run {!r}".format(reps, train_run)
                    )

                train_run._configure_layers_proxy("decode")
                available_envs = self.layers.io.get_available_environments()

                unavailable_envs = [env for env in envs if env not in available_envs]
                if unavailable_envs:
                    msg = f"requested unavailable envs: {unavailable_envs}"
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                envs = envs if envs else available_envs
                if not envs:
                    self.logger.error("no envs specified or detected")
                    raise RuntimeError("no envs specified or detected")

                for env in envs:
                    self.logger.info(f"\\_ analysing performance for env: {env}")

                    ingest_args["io"] = ingest_args.get("io", {})
                    ingest_args["io"]["env"] = env
                    for rep in reps:
                        self.logger.debug(f"\\_ analysing performance for rep: {rep}")

                        ingest_args["io"]["rep"] = rep
                        try:
                            analysis_data_holder += self.analyse_run_pair(
                                train_phase, eval_phase, idx, ingest_args, standalone=False
                            )
                        except (TypeError, ValueError) as e:
                            self.logger.error(
                                (
                                    "Analysing run pair for phase mapping: {}, idx: {}, "
                                    "env: {}, rep: {}, failed! Error: {}."
                                ).format((train_phase, eval_phase), idx, env, rep, e)
                            )

        columns = [
            "phase",
            "trained_with",
            "environment",
            "bit_rate",
            "symbol_rate",
            "repetition",
            "bit_error",
            "symbol_error",
            "matcher",
            "variable",
        ]
        analysis_data = pd.DataFrame(analysis_data_holder).reindex(columns=columns)

        if self.performance_metrics is None:
            self.performance_metrics = analysis_data
        else:
            self.performance_metrics = self.performance_metrics.merge(
                analysis_data, how="outer"
            )

        return analysis_data

    @property
    def performance_metrics(self) -> pd.DataFrame:
        return getattr(self, "_performance_metrics", None)

    @performance_metrics.setter
    def performance_metrics(self, value):
        if isinstance(value, (pd.DataFrame, type(None))):
            setattr(self, "_performance_metrics", value)
        else:
            raise TypeError(f"Wrong type supplied to performance_metrics: {type(value)}")

    @performance_metrics.deleter
    def performance_metrics(self):
        if hasattr(self, "_performance_metrics"):
            delattr(self, "_performance_metrics")

    def aggregate_performance_metrics(self, function: t.Callable = np.mean) -> pd.DataFrame:
        """Aggregatas the performance metrics using a function

        Args:
            function (t.Callable, optional): The aggregator function, takes a Series as an argument

        Returns:
            pd.DataFrame: A DataFrame similar to performance_metrics, without repetitions

        Raises:
            TypeError: Wrong type is provided for the function
            ValueError: Performance metrics not available
        """
        if not isinstance(function, t.Callable):
            raise TypeError("'function' must be a callable")
        if self.performance_metrics is None:
            raise ValueError("performance metrics are not yet available")

        all_columns = self.performance_metrics.columns
        group_columns = [c for c in all_columns if "_error" not in c and c != "repetition"]
        error_columns = [c for c in all_columns if "_error" in c]

        return self.performance_metrics.groupby(group_columns, as_index=False)[
            error_columns
        ].aggregate(function)


"""
PerformanceRun
--------------

Performance runs are considered to be immutable during the Experiment execution. Once
the configuration is provided, the instance is 'frozen' and the config should not be
updated.

Values that are always provided to layers at runtime:
- From a Run's config: phase, bit_count, symbol_rate, repetitions,
- Obtained from a Run's config and the parent: bit_rate.

Values in the parent Experiment can be easily accessed through the parent proxy:
`self.parent`.
"""


class PerformanceRun(
    Run,
    StreamHandler,
    Ibitstream, Isymstream, Ilnestream, Irdpstream, Irawstream,
    Obitstream, Osymstream, Olnestream, Ordpstream, Orawstream, Oschedules,
    serialise_save=[
        "o_bitstream",
        "o_symstream",
        "o_lnestream",
        "o_rdpstream",
        "o_rawstream",
        "o_schedules",],
    serialise_ignore=[
        "i_rawstream",
        "i_rdpstream",
        "i_lnestream",
        "i_symstream",
        "i_bitstream",
    ],
    parent=PerformanceExperiment,
):
    @property
    def path(self):
        formatted_directory = run_path_formatter(self.config.phase, self.config.symbol_rate_id)
        return Path.joinpath(self.parent.path, formatted_directory)

    @classmethod
    def read(cls, path: Path, parent: t.Optional[object] = None) -> object:
        instance = super().read(path, parent)
        instance.load_data()
        return instance

    def write(self) -> None:
        """Serialises the PerformanceRun

        In addition to the base class'es `write`, the PerformanceRun also writes an
        archive with output streams and writes the schedules to *.sched files.
        """
        super().write()
        self.save_data()
        self.write_schedules()

    @property
    def required_config_keys(self):
        return ["phase", "bit_count", "symbol_rate", "repetitions"]

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
            assert self.configured, "Bit count must be available in the configuration"
            return self.config.bit_count

    def _configure_layers_proxy(self, which: str, **kwargs) -> None:
        """Configure runtime-configurable layers

        Layers are configured with own config (phase, bit_count, symbol_rate), and
        optionally with values in kwargs. The layer keys must match (e.g. 'io', 'src')!

        Since the config contains only the symbol_rate, the bit rate is added too.

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

        config = {layer: {} for layer in layers}
        for layer in config:
            config[layer].update(self.config)
            config[layer].update(
                bit_rate=self.parent.layers.src.symrate_to_bitrate(self.config.symbol_rate),
                subsymbol_rate=self.parent.layers.lne.symrate_to_subsymrate(
                    self.config.symbol_rate
                ),
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

        self.logger.debug("<--- digesting begun! --->")
        self.logger.debug("producing bitstream")
        self.o_bitstream = kwargs.get("bitstream", self.make_random_intarray())
        self.logger.debug("producing symstream <- encoding bitstream")
        self.o_symstream = self.parent.layers.src.encode(self.o_bitstream, skip_checks)

        # Handle the case when the source layer performs some padding
        self.o_bitstream = self.parent.layers.src.decode(
            self.o_symstream, skip_checks
        ).flatten()

        self.logger.debug("producing lnestream <- encoding symstream")
        self.o_lnestream = self.parent.layers.lne.encode(self.o_symstream, skip_checks)
        self.logger.debug("producing rdpstream <- encoding lnestream")
        self.o_rdpstream = self.parent.layers.rdp.encode(self.o_lnestream, skip_checks)
        self.logger.debug("producing rawstream <- encoding rdpstream")
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
        self.logger.debug("producing rdpstream <- choosing data and preprocessing rawstream")
        self.i_rdpstream = self.parent.layers.io.decode(self.i_rawstream, skip_checks)
        self.logger.debug("producing lnestream <- preprocessing rdpstream")
        self.i_lnestream = self.parent.layers.rdp.decode(self.i_rdpstream, skip_checks)
        self.logger.debug("producing symstream <- decoding lnestream")
        self.i_symstream = self.parent.layers.lne.decode(self.i_lnestream, skip_checks)
        self.logger.debug("producing bitstream <- decoding symstream")
        self.i_bitstream = self.parent.layers.src.decode(self.i_symstream, skip_checks)
        self.logger.debug("<--- ingesting completed! --->")
        self.collect_intermediates(self.parent.layers)

    def estimated_duration(self, env=None) -> t.Optional[float]:
        """Get the estimated duration of this Run's execution

        Returns:
            t.Optional[float]: the duration in seconds, or None if not digested
        """
        if self.digested:
            bitrate = self.parent.layers.src.symrate_to_bitrate(self.config.symbol_rate)
            return len(self.o_bitstream) / bitrate
        else:
            return None
