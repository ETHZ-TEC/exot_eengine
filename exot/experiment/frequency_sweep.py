"""Capacity bound experiment"""
import copy
import typing as t
from functools import partial, reduce
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate

from exot.exceptions import *
from exot.util.attributedict import AttributeDict
from exot.util.misc import safe_eval, validate_helper
from exot.util.scinum import get_nearest_index, get_welch
from exot.util.wrangle import Matcher

from ._base import Experiment, Run
from ._mixins import *

__all__ = "FrequencySweepExperiment"

global COLUMN_FREQ
global COLUMN_PSD

COLUMN_FREQ = "frequency:fft::Hz"
#COLUMN_PSD = "power_spectral_density:*/Hz"
COLUMN_PSD = "power_spectral_density:K²/Hz"


class FrequencySweepExperiment(
    Experiment, serialise_save=["spectra", "p0"], type=Experiment.Type.FrequencySweep
):
    """Stub class for a capacity experiment
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = FrequencySweepRun

    @staticmethod
    def required_layers():
        return ["rdp", "io"]

    def write(self):
        super().write()
        self.save_data_bundled(prefix="data")

    @classmethod
    def read(cls, *args, **kwargs) -> object:
        instance = super().read(*args, **kwargs)
        instance.load_data_bundled(prefix="data")

        return instance

    def validate(self):
        """Validate experiment configuration"""

        # In addition to the validation performed in the parent class...
        super().validate()

        validate = partial(validate_helper, self.config, msg="FrequencySweep")

        # ...verify experiment phases configuration
        for k in self.config.EXPERIMENT.PHASES:
            validate(("EXPERIMENT", "PHASES", k), AttributeDict)
            validate(("EXPERIMENT", "PHASES", k, "length_seconds"), int)
            validate(("EXPERIMENT", "PHASES", k, "frequencies"), str, list)
            validate(("EXPERIMENT", "PHASES", k, "repetitions"), int)
            validate(("EXPERIMENT", "PHASES", k, "signal"), list)

        # ... verify general zone/platform settings
        validate(("EXPERIMENT", "GENERAL"), AttributeDict)
        validate(("EXPERIMENT", "GENERAL", "latency"), int)
        validate(("EXPERIMENT", "GENERAL", "fan"), bool, str, list)
        validate(("EXPERIMENT", "GENERAL", "governors"), str, list)
        validate(("EXPERIMENT", "GENERAL", "frequencies"), str, float, list)
        validate(("EXPERIMENT", "GENERAL", "sampling_period"), float)

    def generate(self):
        assert self.configured, "Experiment must be configured before generating"
        assert self.bootstrapped, "Experiment must be bootstrapped before generating"

        self.phases = {tag: {} for tag in self.config.EXPERIMENT.PHASES}
        self.estimated_duration = {tag: 0.0 for tag in self.config.EXPERIMENT.PHASES}

        for phase, values in self.config.EXPERIMENT.PHASES.items():
            frequencies = values["frequencies"]

            if isinstance(frequencies, str):
                self.logger.info(
                    f"frequencies in phase {phase!r} given as a str, will be evaluated"
                )
                frequencies = safe_eval(frequencies)

            if not isinstance(frequencies, (t.List, np.ndarray)):
                raise GenerateTypeAssertion("frequencies must be a list")

            types = set(type(x) for x in frequencies)
            if not all(issubclass(_, (float, np.float, int, np.integer)) for _ in types):
                raise GenerateTypeAssertion(
                    f"frequencies should only be int's or float's, but were: {types}"
                )

            frequency_id = 0
            for frequency in frequencies:
                self.logger.debug(f"generating run for phase: {phase}, frequency: {frequency}")
                self.phases[phase][frequency_id] = FrequencySweepRun(
                    config=AttributeDict(
                        phase=phase,
                        length_seconds=values["length_seconds"],
                        frequency=frequency,
                        frequency_id=frequency_id,
                        repetitions=values["repetitions"],
                    ),
                    parent=self,
                )

                # Perform all encodings
                self.phases[phase][frequency_id].digest()
                frequency_id += 1
            self.estimated_duration[phase] = (
                values["length_seconds"] + self.estimated_delays_duration
            ) * len(self.phases[phase])

    def _get_peak(self, spectrum: pd.DataFrame, phase: str, f_id: int) -> t.Dict:
        """
        Gets the spectrum peaks in small frequency intervals

        Args:
            spectrum (pd.DataFrame): The spectrum
            phase (str): The experiment phase
            f_id (int): The frequency id

        Returns:
            t.Dict: A mapping with frequency and power spectral density
        """
        freqs = self.config.EXPERIMENT.PHASES[phase]["frequencies"]

        freq_prev = freqs[f_id - 1] if f_id > 0 else 0.0
        freq_curr = freqs[f_id]
        freq_next = freqs[f_id + 1] if f_id < len(freqs) - 1 else freqs[-1] + 1

        try:
            # search window
            f_low = freq_prev + float(freq_curr - freq_prev) / 2
            f_hig = freq_next - float(freq_next - freq_curr) / 2

            idx = spectrum[(spectrum[COLUMN_FREQ] > f_low) & (spectrum[COLUMN_FREQ] < f_hig)][
                COLUMN_PSD
            ].idxmax()
            spectrum_max = spectrum.loc[idx].to_frame().T
        except ValueError:
            # closest match
            idx = get_nearest_index(spectrum[COLUMN_FREQ], freq_curr)
            spectrum_max = spectrum.loc[idx].to_frame().T

        return {
            "variable": str(spectrum_max["variable"].item()),
            COLUMN_FREQ: float(spectrum_max[COLUMN_FREQ].item()),
            COLUMN_PSD: float(spectrum_max[COLUMN_PSD].item()),
        }

    def generate_spectra(
        self,
        *,
        phases: t.List[str] = [],
        envs: t.List[str] = [],
        reps: t.List[int] = [],
        **kwargs,
    ) -> None:
        """This function generates following channel:
             * Shh ... Channel Spectrum
             * Sqq ... Noise Spectrum (if frequency=0 has been evaluated, otherwise 0)
        """
        if not isinstance(phases, t.List):
            raise TypeError(f"'phases' argument must be a list, got: {type(phases)}")
        if not isinstance(envs, t.List):
            raise TypeError(f"'envs' argument must be a list, got: {type(envs)}")
        if not isinstance(reps, t.List):
            raise TypeError(f"'reps' argument must be a list, got: {type(reps)}")
        if not all([isinstance(_, int) for _ in reps]):
            raise ValueError(f"'reps' argument must contain only integers")
        if not all([isinstance(_, str) for _ in phases]):
            raise ValueError("'phases' argument must contain only strings")

        invalid_phases = [_ for _ in phases if _ is not None and _ not in self.phases]
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

        description = "{}_{}".format(matcher[0][0]._quantity, matcher[0][0]._method)
        ingest_args = kwargs.copy()
        ingest_args["io"] = ingest_args.get("io", {})
        ingest_args["io"].update(matcher=matcher)

        df_holder = []
        p0_holder = []

        for phase in phases:
            for env in envs:
                self.logger.info(f"\\_ generating spectra for env: {env}")
                ingest_args["io"] = ingest_args.get("io", {})
                ingest_args["io"]["env"] = env

                for rep in reps:
                    self.logger.debug(f"\\_ analysing performance for rep: {rep}")
                    ingest_args["io"]["rep"] = rep

                    df, p0 = self._inner_generate_spectra(
                        phase=phase, env=env, rep=rep, **ingest_args
                    )
                    df_holder.append(df)
                    p0_holder.append(dict(phase=phase, environment=env, repetition=rep, p0=p0))

        columns = [
            "phase",
            "environment",
            "repetition",
            "spectrum",
            COLUMN_FREQ,
            COLUMN_PSD,
            "matcher",
            "variable",
        ]

        analysis_spectra = pd.concat(df_holder, ignore_index=True).reindex(columns=columns)
        analysis_p0 = pd.DataFrame(p0_holder).reindex(
            columns=["phase", "environment", "repetition", "p0"]
        )

        if self.spectra is None:
            self.spectra = analysis_spectra
        else:
            self.spectra = self.spectra.merge(analysis_spectra, how="outer")

        if self.p0 is None:
            self.p0 = analysis_p0
        else:
            self.p0 = self.p0.merge(analysis_p0, how="outer")

        return analysis_spectra

    def _inner_generate_spectra(self, phase, env, rep, **kwargs) -> None:
        Sxx = pd.DataFrame()
        Syy = pd.DataFrame()
        Shh = pd.DataFrame()
        p0 = list()

        for f_id in self.phases[phase]:
            cur_run = self.phases[phase][f_id]

            all_reps = np.arange(cur_run.config.repetitions).tolist()
            if rep not in all_reps:
                raise ValueError(
                    "provided reps ({}) invalid for run {!r}".format(reps, train_run)
                )

            cur_run._configure_layers_proxy("decode")
            if env not in self.layers.io.get_available_environments():
                msg = f"requested unavailable env: {env}"
                self.logger.error(msg)
                raise RuntimeError(msg)

            # Make sure the current run is ingested with the right settings
            cur_run.ingest(**kwargs)

            if cur_run.config.frequency == 0:
                # If this is the noise run, add the spectrum as the noise spectrum
                Sqq = copy.deepcopy(cur_run.i_fspectrum)
            else:
                # Get the peak of the spectrum
                Sxx = Sxx.append(
                    self._get_peak(cur_run.o_fspectrum, phase, f_id), ignore_index=True
                )
                Syy = Syy.append(
                    self._get_peak(cur_run.i_fspectrum, phase, f_id), ignore_index=True
                )
                Shh = Shh.append(
                    {
                        "variable": str(Syy["variable"].iloc[-1]),
                        COLUMN_FREQ: Syy[COLUMN_FREQ].iloc[-1],
                        COLUMN_PSD: Syy[COLUMN_PSD].iloc[-1]/Sxx[COLUMN_PSD].iloc[-1],
                    },
                    ignore_index=True,
                )
                p0.append(cur_run.o_p0)

        # Crop and resample all the spectra to the same shape
        if Sqq is not None:
            f_mask = (
                max([Sxx[COLUMN_FREQ].min(), Syy[COLUMN_FREQ].min(), Shh[COLUMN_FREQ].min()])
                <= Sqq[COLUMN_FREQ]
            ) & (
                min([Sxx[COLUMN_FREQ].max(), Syy[COLUMN_FREQ].max(), Shh[COLUMN_FREQ].max()])
                >= Sqq[COLUMN_FREQ]
            )

            def interpolate_spectrum(f_new, spectrum):
                interpolation = interpolate.interp1d(
                    spectrum[COLUMN_FREQ].values, spectrum[COLUMN_PSD].values
                )
                df = pd.DataFrame({COLUMN_FREQ: f_new, COLUMN_PSD: interpolation(f_new)})
                df["variable"] = spectrum["variable"].iloc[0]
                return df

            Sxx = interpolate_spectrum(Sqq[COLUMN_FREQ][f_mask].values, Sxx)
            Syy = interpolate_spectrum(Sqq[COLUMN_FREQ][f_mask].values, Syy)
            Shh = interpolate_spectrum(Sqq[COLUMN_FREQ][f_mask].values, Shh)
            Sqq = Sqq[f_mask]

        Sxx["variable"] = Syy["variable"]

        Sxx["environment"] = env
        Syy["environment"] = env
        Shh["environment"] = env
        Sqq["environment"] = env

        Sxx["repetition"] = rep
        Syy["repetition"] = rep
        Shh["repetition"] = rep
        Sqq["repetition"] = rep

        Sxx["phase"] = phase
        Syy["phase"] = phase
        Shh["phase"] = phase
        Sqq["phase"] = phase

        Sxx["matcher"] = repr(kwargs["io"]["matcher"])
        Syy["matcher"] = repr(kwargs["io"]["matcher"])
        Shh["matcher"] = repr(kwargs["io"]["matcher"])
        Sqq["matcher"] = repr(kwargs["io"]["matcher"])

        Sxx["spectrum"] = "Sxx"
        Syy["spectrum"] = "Syy"
        Shh["spectrum"] = "Shh"
        Sqq["spectrum"] = "Sqq"

        return pd.concat([Sxx, Syy, Shh, Sqq]), np.array(p0).max()

    @property
    def spectra(self):
        return getattr(self, "_spectra", None)

    @spectra.setter
    def spectra(self, value):
        if not isinstance(value, (pd.DataFrame, type(None))):
            raise TypeError()
        else:
            self._spectra = value

    @spectra.deleter
    def spectra(self):
        if hasattr(self, "_spectra"):
            delattr(self, "_spectra")

    @property
    def p0(self):
        return getattr(self, "_p0", None)

    @p0.setter
    def p0(self, value):
        if isinstance(value, (pd.DataFrame, type(None))):
            setattr(self, "_p0", value)

    @p0.deleter
    def p0(self):
        if hasattr(self, "_p0"):
            delattr(self, "_p0")

    def spectrum_as_matrix(
        self, spectrum: str, phase: str, env: str, rep: t.Union[t.List[int], int] = 0
    ):
        if self.spectra is None:
            return np.array([])
        else:
            rep = rep if isinstance(rep, t.List) else [rep]
            combine_and = lambda *cond: reduce(np.logical_and, cond)
            query = combine_and(
                self.spectra.phase == phase,
                self.spectra.spectrum == spectrum,
                self.spectra.environment == env,
                np.isin(self.spectra.repetition, rep),
            )
            return self.spectra[query][[COLUMN_FREQ, COLUMN_PSD]].values


"""
FrequencySweepRun
--------------

FrequencySweep runs are considered to be immutable during the Experiment execution. Once
the configuration is provided, the instance is 'frozen' and the config should not be
updated.

Values that are always provided to layers at runtime:
- From a Run's config: phase, length_seconds, frequency, repetitions,
- Obtained from a Run's config and the parent: bit_rate.

Values in the parent Experiment can be easily accessed through the parent proxy:
`self.parent`.
"""


class FrequencySweepRun(
    Run,
    StreamHandler,
    Ilnestream, Irdpstream, Irawstream,
    Olnestream, Ordpstream, Orawstream, Oschedules,
    serialise_save=[
        "o_lnestream",
        "o_rdpstream",
        "o_rawstream",
        "o_schedules",],
    serialise_ignore=[
        "o_fspectrum",
        "i_rawstream",
        "i_rdpstream",
        "i_lnestream",
        "i_fspectrum",
    ],
    parent=FrequencySweepExperiment,
):
    @property
    def identifier(self):
        return self.config.frequency_id

    @classmethod
    def read(cls, path: Path, parent: t.Optional[object] = None) -> object:
        instance = super().read(path, parent)
        instance.load_data_bundled()
        return instance

    def write(self) -> None:
        """Serialises the FrequencySweepRun

        In addition to the base class'es `write`, the FrequencySweepRun also writes an
        archive with output streams and writes the schedules to *.sched files.
        """
        super().write()
        self.save_data_bundled()
        self.write_schedules()

    @property
    def required_config_keys(self):
        return ["phase", "length_seconds", "frequency", "repetitions"]

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
            return self.config.length_seconds

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
            if "carrier" in self.parent.channel.signal.keys():
                # Use the carrier signal as base for the sweep
                subsymbol_count = len(self.parent.channel.signal["carrier"])
            else:
                # Use alternating 0 and 1 as base for sweep
                subsymbol_count = 2
            if self.config.frequency == 0:
                frequency = (
                    0.5
                )  # Zero is not allowed, therefore set 0.5 to get a DC trace of the desired length
            else:
                frequency = self.config.frequency
            _[layer].update(
                bit_rate=frequency,
                symbol_rate=frequency,
                subsymbol_rate=(frequency * subsymbol_count),
                environments_apps_zones=self.parent.environments_apps_zones,
                path=self.path,
            )

            if which == "decode" and self.digested:
                _[layer].update(**self.o_streams)

            if layer in kwargs:
                _[layer].update(**kwargs.pop(layer))

            self.logger.debug(f"configuring {which} of layer {layer}")

        configurator(**_)

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
        self.logger.debug("producing lnestream")
        num_periods = int(np.ceil(self.config["length_seconds"] * self.config["frequency"]))
        if self.config.frequency == 0:
            self.o_lnestream = self.make_constant_intarray(0, 2 * num_periods)
        else:
            self.o_lnestream = self.make_repeated_intarray(
                self.parent.config.EXPERIMENT.PHASES[self.config.phase].signal,
                len(self.parent.config.EXPERIMENT.PHASES[self.config.phase].signal)
                * num_periods,
            )
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

        # Calculate the spectra of the trace
        schedule_tag = self.parent.environments[kwargs.get("io")["env"]][
            self.parent.config.ENVIRONMENTS[kwargs.get("io")["env"]].APPS.src.zone
        ]["schedule_tag"]


        def resample_o_lnestream():
            data = np.vstack(
                [np.hstack([0, self.o_rdpstream["timestamp"].values.cumsum()]), np.hstack([self.o_lnestream[0], self.o_lnestream])]
            ).transpose()
            interpolation = interpolate.interp1d(
                data[:, 0], data[:, 1], kind="next"
            )
            #x_new = np.arange(
            #    data[0, 0],
            #    data[:, 0].cumsum()[-1],
            #    self.parent.config.EXPERIMENT.GENERAL.sampling_period,
            #)
            x_new = self.i_rdpstream.iloc[:,0].to_numpy()

            try:
                data_new = interpolation(x_new)
            except ValueError:
                x_new = x_new[x_new < data[-1, 0]]
                data_new = interpolation(x_new)

            data_new = data_new - (
                abs(self.o_lnestream.max() - self.o_lnestream.min()) / 2
            )  # Make DC Free
            return pd.DataFrame({"timestamp": x_new, "data": data_new})

        self.o_fspectrum = get_welch(
            resample_o_lnestream(), kwargs.get("window_size"), timescale=1
        )
        self.o_fspectrum = self.o_fspectrum.rename(columns={"value": COLUMN_PSD})
        self.o_p0 = (
            self.o_fspectrum[COLUMN_PSD].values
            * np.diff(self.o_fspectrum[COLUMN_FREQ].values).mean()
        ).sum()
        self.i_fspectrum = get_welch(self.i_rdpstream, kwargs.get("window_size"), timescale=1)
        self.i_fspectrum = self.i_fspectrum.rename(columns={"value": COLUMN_PSD})
        self.collect_intermediates(self.parent.layers)

        self.logger.debug("<--- ingesting completed! --->")

    def estimated_duration(self, env=None) -> t.Optional[float]:
        return self.config.length_seconds

    # Input spectrum from read values
    @property
    def i_fspectrum(self):
        return getattr(self, "_i_fspectrum", None)

    @i_fspectrum.setter
    def i_fspectrum(self, value):
        if isinstance(
            value, (pd.core.frame.DataFrame, np.ndarray)
        ):  # f"value is of type {type(value)}, should be bitarray or np.ndarray"
            setattr(self, "_i_fspectrum", value)

    @i_fspectrum.deleter
    def i_fspectrum(self):
        if hasattr(self, "_i_fspectrum"):
            delattr(self, "_i_fspectrum")

    @property
    def o_fspectrum(self):
        return getattr(self, "_o_fspectrum", None)

    @o_fspectrum.setter
    def o_fspectrum(self, value):
        if isinstance(
            value, (pd.core.frame.DataFrame, np.ndarray)
        ):  # f"value is of type {type(value)}, should be bitarray or np.ndarray"
            setattr(self, "_o_fspectrum", value)

    @o_fspectrum.deleter
    def o_fspectrum(self):
        if hasattr(self, "_o_fspectrum"):
            delattr(self, "_o_fspectrum")

    # Power Cap derived from input values
    @property
    def o_p0(self):
        return getattr(self, "_o_p0", None)

    @o_p0.setter
    def o_p0(self, value):
        if isinstance(
            value, (float)
        ):  # f"value is of type {type(value)}, should be bitarray or np.ndarray"
            setattr(self, "_o_p0", value)

    @o_p0.deleter
    def o_p0(self):
        if hasattr(self, "_o_p0"):
            delattr(self, "_o_p0")
