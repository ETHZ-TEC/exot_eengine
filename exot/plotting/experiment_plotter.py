"""Plotting support for experimental experiments"""

import typing as t

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from exot.experiment._base import Experiment
from exot.experiment.frequency_sweep import FrequencySweepExperiment
from exot.experiment.performance import PerformanceExperiment
from exot.util.wrangle import filter_data, get_unique_column_combinations

from ._base import Plotter


class ExperimentPlotter(Plotter):
    """Plotting support for experiment instances

    Attributes:
        PLOT_FILENAMES (dict): Default filenames for supported plots
    """

    PLOT_FILENAMES = {}

    def __init__(self, experiment, *args, **kwargs):

        self._experiment = experiment
        if "save_path" not in kwargs:
            kwargs["save_path"] = self.experiment.path
        super().__init__(*args, **kwargs)

    @property
    def experiment(self) -> Experiment:
        return self._experiment


class FrequencySweepExperimentPlotter(ExperimentPlotter):

    """Plotting support for FrequencySweepExperiment instances
    """

    def __init__(self, experiment: Experiment, *args, **kwargs):
        if not isinstance(experiment, FrequencySweepExperiment):
            raise TypeError(
                "FrequencySweepExperimentPlotter accepts only FrequencySweepExperiment instances"
            )

        if experiment.spectra is None:
            raise ValueError(
                "The plotter requires an experiment with generated channel spectra"
            )

        if experiment.spectra.empty:
            raise ValueError("The experiment's channel spectra data frame is empty")

        super().__init__(experiment, *args, **kwargs)

    def plot_channel_spectra(self, selector={}, **kwargs):
        if not isinstance(selector, (t.Dict)):
            raise TypeError("selector must be a dictionary")

        SUBPLOT_HEIGHT = 4

        data = filter_data(self.experiment.spectra, **selector)
        phase_env = get_unique_column_combinations(data, ["phase", "environment"])

        f, axes = plt.subplots(
            phase_env.size,
            1,
            figsize=(self._width, phase_env.size * SUBPLOT_HEIGHT),
            dpi=self._screen_dpi,
            sharex=False,
            sharey=False,
            squeeze=False,
        )

        for idx, axis in enumerate(axes.ravel()):
            sns.lineplot(
                x="frequency:fft::Hz",
                y="power_spectral_density:K²/Hz",
                hue="spectrum",
                alpha=0.8,
                data=filter_data(data, **dict(zip(["phase", "environment"], phase_env[idx]))),
                ax=axis,
            )
            axis.set_xscale("log")
            axis.grid()
            axis.grid(which="minor", linestyle=":", alpha=0.5)
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel("Power Spectral Density (K²/Hz)")
            axis.set_title("Spectra for phase: {}, environment: {}".format(*phase_env[idx]))

        sns.despine(f)
        f.suptitle("Channel spectra")
        f.tight_layout()
        f.subplots_adjust(top=0.85)
        self._plot_save_helper(f, **kwargs)


class PerformanceExperimentPlotter(ExperimentPlotter):

    """Plotting support for PerformanceExperiment instances
    """

    def __init__(self, experiment: Experiment, *args, **kwargs):
        if not isinstance(experiment, PerformanceExperiment):
            raise TypeError(
                "PerformanceExperimentPlotter accepts only PerformanceExperiment instances"
            )

        if experiment.performance_metrics is None:
            raise ValueError(
                "The plotter requires an experiment with calculated performance metrics"
            )

        if experiment.performance_metrics.empty:
            raise ValueError("The experiment's performance metrics data frame is empty")

        super().__init__(experiment, *args, **kwargs)

    def plot_performance_metrics(
        self, bit_or_symbol="bit", selector={}, *, estimator=np.mean, **kwargs
    ):
        if not isinstance(selector, (t.Dict)):
            raise TypeError("selector must be a dictionary")
        if not isinstance(estimator, (str, t.Callable)):
            raise TypeError(f"estimator must be a string or a callable, got: {type(estimator)}")
        if bit_or_symbol not in ["bit", "symbol"]:
            raise ValueError(
                f"'bit_or_symbol' must be either 'bit' or 'symbol', got: {str(bit_or_symbol)}"
            )

        SUBPLOT_HEIGHT = 4

        data = filter_data(self.experiment.performance_metrics, **selector)
        cc = get_unique_column_combinations(data, ["phase", "trained_with", "environment"])

        train_phase_t_env = [_ for _ in cc if _[1] not in [_[0] for _ in cc]]
        eval_phase_t_env = [_ for _ in cc if _[1] in [_[0] for _ in cc]]
        phase_t_env = eval_phase_t_env if eval_phase_t_env else train_phase_t_env
        phase_env = [(_[0], _[2]) for _ in phase_t_env]
        phases = list(set([_[0] for _ in phase_env]))

        if not phases:
            raise ValueError("no phases to plot")

        f, axes = plt.subplots(
            len(phases),
            1,
            figsize=(self._width, len(phases) * SUBPLOT_HEIGHT),
            dpi=self._screen_dpi,
            sharex=False,
            sharey=False,
            squeeze=False,
        )

        for idx, axis in enumerate(axes.ravel()):
            plot_data = filter_data(data, **dict(zip(["phase", "environment"], phase_env[idx])))

            sns.lineplot(
                x="bit_rate" if bit_or_symbol == "bit" else "symbol_rate",
                y="bit_error" if bit_or_symbol == "bit" else "symbol_error",
                hue="environment",
                data=plot_data,
                err_style="bars",
                err_kws={"capsize": 5},
                estimator=estimator,
                ci="sd",
                sort=True,
                markers=True,
                linewidth=1.0,
                ax=axis,
            )

            axis.set_ylim(-0.05, 0.70 if bit_or_symbol == "bit" else 1.0)
            axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
            axis.grid(axis="y")
            axis.set_xlabel(
                "Bit rate (b/s)" if bit_or_symbol == "bit" else "Symbol rate (sym/s)"
            )
            axis.set_ylabel("Error rate")
            axis.set_title(
                "Performance metrics for phase: '{}', trained with: '{}'".format(
                    phases[idx], phase_t_env[idx][1]
                )
            )
            axis.legend(loc="upper left", frameon=False)

        sns.despine(f)
        f.suptitle("Performance metrics")
        f.tight_layout()
        f.subplots_adjust(top=0.85)
        self._plot_save_helper(f, **kwargs)
