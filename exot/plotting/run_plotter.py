"""Plotting support for experimental runs"""

import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns

from exot.experiment._base import Run
from exot.experiment.frequency_sweep import FrequencySweepRun
from exot.experiment.performance import PerformanceRun
from exot.util.attributedict import AttributeDict
from exot.util.plotting import add_spine, remove_spine, rugplot
from exot.util.scinum import is_fitted, unpack_array

from ._base import Plotter

__all__ = ("RunPlotter", "PerformanceRunPlotter", "FrequencySweepRunPlotter")


class RunPlotter(Plotter):
    """Plotting support for Run instances

    Attributes:
        PLOT_FILENAMES (dict): Default filenames for supported plots
    """

    PLOT_FILENAMES = {}

    def __init__(self, run, *args, **kwargs):
        if not run.digested and run.ingested:
            raise ValueError("Plotter requires ingested and digested Run's")
        self._run = run
        if "save_path" not in kwargs:
            kwargs["save_path"] = self.run.path
        super().__init__(*args, **kwargs)

    @property
    def run(self) -> Run:
        return self._run

    def _raw_plot_helper(
        self,
        source: pandas.DataFrame,
        start: t.Optional[float] = None,
        end: t.Optional[float] = None,
        dim_count: t.Optional[int] = None,
    ):
        SUBPLOT_HEIGHT = 1  # inches

        timestamps = source.iloc[:, 0]
        data = source.iloc[:, 1:]
        dims = data.shape[1]
        dims_to_plot = data.shape[1]

        if dim_count is not None:
            if dim_count > dims:
                raise ValueError(f"dim_count ({dim_count}) > dims ({dims})")

            dims_to_plot = dim_count

        start = start if start else timestamps.iloc[0]
        end = end if end else timestamps.iloc[-1]
        interval = timestamps.between(start, end)

        timestamps = timestamps[interval]
        data = data[interval]

        # Create subplots: 3 columns, ndim rows
        f, axes = plt.subplots(
            dims_to_plot,
            1,
            figsize=(self._width, 1 + dims_to_plot * SUBPLOT_HEIGHT),
            dpi=self._screen_dpi,
            sharex="col",
            sharey="col",
            squeeze=False,
        )

        lower_ylim, upper_ylim = np.quantile(data, [0.01, 0.99])

        for i, axis in enumerate(axes[:, 0]):
            axis.plot(
                timestamps, data.iloc[:, i], marker="+", markersize=2, linewidth=0.5, alpha=0.5
            )
            axis.set_ylim(0.975 * lower_ylim, 1.025 * upper_ylim)
            axis.get_xaxis().get_major_formatter().set_useOffset(False)
            axis.set_xlim(timestamps.iloc[0], timestamps.iloc[-1])
            axis.set_ylabel("{}\n{}\n{} ({})".format(*data.columns[i].split(":")), color="gray")

        annotations = None
        if 'io' in self.run.intermediates:
            if np.isclose(timestamps.iloc[0], 0.0) and "src_log" in self.run.intermediates.io:
                annotations = self.run.intermediates.io.src_log.iloc[[0, -1], 0]
            elif (
                not np.isclose(timestamps.iloc[0], 0.0)
            ) and "raw_src_log" in self.run.intermediates.io:
                annotations = self.run.intermediates.io.raw_src_log.iloc[[0, -1], 0]

        if annotations is not None:
            for axis in axes.ravel():
                axis.vlines(
                    annotations,
                    0,
                    1,
                    transform=axis.get_xaxis_transform(),
                    linewidth=1.0,
                    linestyle="--",
                )

        sns.despine()

        axes[-1, 0].set_xlabel(source.columns[0], color="gray")

        _title = (
            "Processed raw data stream"
            if np.isclose(timestamps.iloc[0], 0.0)
            else "Raw data stream"
        )

        if dims_to_plot != dims:
            _title += " [{} of {} dimensions]".format(dims_to_plot, dims)

        plt.suptitle(_title, y=1.01, verticalalignment="bottom")
        f.tight_layout()

        return f

    def plot_rawstream(
        self,
        start: t.Optional[float] = None,
        end: t.Optional[float] = None,
        dim_count: t.Optional[int] = None,
        **kwargs,
    ):
        f = self._raw_plot_helper(
            self.run.i_rawstream, start=start, end=end, dim_count=dim_count
        )
        self._plot_save_helper(f, **kwargs)

    def plot_rdpstream(
        self,
        start: t.Optional[float] = None,
        end: t.Optional[float] = None,
        dim_count: t.Optional[int] = None,
        **kwargs,
    ):
        f = self._raw_plot_helper(
            self.run.i_rdpstream, start=start, end=end, dim_count=dim_count
        )
        self._plot_save_helper(f, **kwargs)


class FrequencySweepRunPlotter(RunPlotter):
    """Plotting support for FrequencySweepRun instances
    Attributes:
        PLOT_FILENAMES (dict): Default filenames for supported plots
    """

    def __init__(self, run: FrequencySweepRun, *args, **kwargs):
        if not isinstance(run, FrequencySweepRun):
            raise TypeError("FrequencySweepRunPlotter accepts only FrequencySweepRun instances")
        super().__init__(run, *args, **kwargs)

    def plot_spectrum(self, window=8192):
        pass


class PerformanceRunPlotter(RunPlotter):
    """Plotting support for PerformanceRun instances

    Attributes:
        PLOT_FILENAMES (dict): Default filenames for supported plots
    """

    def __init__(self, run: PerformanceRun, *args, **kwargs):
        if not isinstance(run, PerformanceRun):
            raise TypeError("PerformanceRunPlotter accepts only PerformanceRun instances")
        super().__init__(run, *args, **kwargs)

    def plot_slicing(
        self, start: int = 0, count: int = 10, dim_count: t.Optional[int] = None, **kwargs
    ):
        SUBPLOT_HEIGHT = 1  # inches

        samples_per_symbol = (
            self.run.i_lnestream.shape[1]
            if self.run.i_lnestream.ndim == 2
            else self.run.i_lnestream.shape[2]
        )
        subsymbol_count = getattr(self.run.parent.layers.lne, "subsymbol_count", 1)

        count = min([count, self.run.i_symstream.size])
        start_idx = start
        start_sample = start * samples_per_symbol
        end_idx = start_idx + count
        end_sample = start_sample + samples_per_symbol * (count)
        selection_idx = slice(start_idx, start_idx + count)
        selection_sample = slice(start_sample, end_sample)

        selection_gt = slice(
            start_idx * subsymbol_count, (start_idx + count + 1) * subsymbol_count
        )
        selection_slicing = self.run.intermediates.rdp.slicing[
            slice(start_idx, start_idx + count + 1)
        ]
        selection_raw = self.run.i_rdpstream.iloc[:, 0].between(
            selection_slicing[0], selection_slicing[-1]
        )

        annotations = None
        if "src_log" in self.run.intermediates.io:
            annotations = self.run.intermediates.io.src_log.iloc[:, 0]
            annotations = annotations[
                annotations.between(0.99 * selection_slicing[0], 1.01 * selection_slicing[-1])
            ]

        # Create plotting data, figures, and plot data
        raw: pandas.DataFrame = self.run.i_rdpstream[selection_raw]
        data: np.ndarray
        gt: np.ndarray = self.run.o_lnestream[selection_gt]

        # Handle 2-d and 3-d data
        if self.run.i_lnestream.ndim == 2:
            dims = 1
            data = np.vstack(self.run.i_lnestream).reshape(
                self.run.i_lnestream.shape[0] * samples_per_symbol, 1
            )[selection_sample]

        elif self.run.i_lnestream.ndim == 3:
            dims = self.run.i_lnestream.shape[1]
            data = (
                self.run.i_lnestream.transpose(1, 0, 2)
                .reshape(dims, self.run.i_lnestream.size // dims)
                .T[selection_sample, :]
            )

        dims_to_plot = dims
        if dim_count is not None:
            if dim_count > dims:
                raise ValueError(f"dim_count ({dim_count}) > dims ({dims})")

            dims_to_plot = dim_count

        # Create subplots: 3 columns, ndim rows
        f, axes = plt.subplots(
            dims_to_plot,
            3,
            figsize=(self._width, 1 + dims_to_plot * SUBPLOT_HEIGHT),
            dpi=self._screen_dpi,
            sharex="col",
            sharey="col",
            squeeze=False,
        )

        if dims_to_plot == 1:
            gt = gt.reshape(-1, 1)
        else:
            # if there are more than one data dimensions, that likely means that we're dealing with
            #  MIMO symbols, which need to be "unpacked".
            gt = np.flip(unpack_array(gt, n=dims), axis=1)

        # Handle printing symbolstreams with -1 saturating values
        if -1 in gt:
            gt = np.vectorize(lambda v: 1 if v == -1 else v)(gt)

        # Plot raw, ground truth, and samples
        for i, axis_group in enumerate(axes):
            axis_group[0].plot(raw.iloc[:, 0], raw.iloc[:, i + 1], alpha=1.0, linestyle="-")

            # gt_t is ground truth timing, same as raw timing
            # gt_d is ground truth "data"

            if subsymbol_count != 1:
                gt_t, gt_d = (
                    np.linspace(selection_slicing[0], selection_slicing[-1], gt.shape[0] - 1),
                    gt[:-1, i],
                )
            else:
                gt_t, gt_d = (selection_slicing, np.array(gt[:, i]))
                gt_t, gt_d = gt_t[: gt_d.size], gt_d[: gt_t.size]

            axis_group[1].plot(gt_t, gt_d, marker=".", drawstyle="steps-post")

            axis_group[2].plot(
                data[:, i],
                alpha=min(1.0, 100 / samples_per_symbol),
                linestyle=":",
                marker="+",
                markersize=2,
                linewidth=0.5,
            )

        sns.despine()

        # Column 1, raw data
        axes[-1, 0].set_ylim(
            0.975 * np.quantile(raw.iloc[:, 1:], 0.01),
            1.025 * np.quantile(raw.iloc[:, 1:], 0.99),
        )
        axes[-1, 0].set_xlabel(raw.columns[0], color="gray")

        for i, axis in enumerate(axes[:, 0].ravel()):
            _ = raw.columns[i + 1].split(":")
            axis.set_ylabel("{}\n{}\n{} ({})".format(*_), color="gray")

        # Column 2, ground truth
        axes[-1, 1].set_ylim(np.nanmin(gt_d) - 0.5, np.nanmax(gt_d) + 0.5)
        axes[-1, 1].set_yticks(np.unique(gt_d))
        axes[-1, 1].set_xlabel(raw.columns[0], color="gray")

        for i, axis in enumerate(axes[:, 1].ravel()):
            axis.set_ylabel("Data[{}]".format(i), color="gray")

        # Column 3, sampled data
        xticks = np.arange(0, samples_per_symbol * (count + 1), samples_per_symbol)
        xticklabels = np.arange(
            start_sample, end_sample + samples_per_symbol, samples_per_symbol
        )
        xlabel = "Sample #"

        axes[-1, 2].set_ylim(0.975 * np.quantile(data, 0.1), 1.025 * np.quantile(data, 0.9))
        axes[-1, 2].set_xticks(xticks)
        axes[-1, 2].set_xticklabels(xticklabels, rotation=45)
        axes[-1, 2].set_xlabel(xlabel, color="gray")

        # Plot symbol boundaries in real time space
        for axis in axes[:, [0, 1]].ravel():
            axis.vlines(
                selection_slicing,
                0,
                1,
                transform=axis.get_xaxis_transform(),
                linewidth=1.0,
                alpha=0.5,
                linestyle="--",
                color="k",
            )

        if annotations is not None:
            for axis in axes[:, [0, 1]].ravel():
                axis.vlines(
                    annotations,
                    0.0,
                    0.75,
                    transform=axis.get_xaxis_transform(),
                    linewidth=1.0,
                    alpha=0.5,
                    linestyle="-.",
                    color="r",
                )

        # Plot symbol boundaries in sample space
        for axis in axes[:, 2].ravel():
            axis.vlines(
                xticks,
                0,
                1,
                transform=axis.get_xaxis_transform(),
                linewidth=1.0,
                alpha=0.5,
                linestyle="--",
                color="k",
            )

        # Align labels
        f.align_ylabels(axes[:, :])
        f.align_xlabels(axes[-1, :])

        # Set titles
        axes[0, 0].set_title("Raw data", fontstyle="italic", color="gray")
        axes[0, 1].set_title("Ground truth", fontstyle="italic", color="gray")
        axes[0, 2].set_title("Sample stream", fontstyle="italic", color="gray")

        _title = "Symbol stream for symbols {} to {}".format(start_idx, start_idx + count)

        if dims_to_plot != dims:
            _title += " [{} of {} dimensions]".format(dims_to_plot, dims)

        plt.suptitle(_title, y=1.01, verticalalignment="bottom")
        f.tight_layout()

        self._plot_save_helper(f, **kwargs)

    def plot_symstream(
        self, start: t.Optional[int] = None, count: t.Optional[int] = 10, **kwargs
    ):
        SUBPLOT_HEIGHT = 1.5  # inches

        start = start if start else 0
        end = start + count if count else len(self.run.o_symstream) - start
        dims = self.run.i_rdpstream.shape[1] - 1

        _out = self.run.o_symstream[slice(start, end)]
        _in = self.run.i_symstream[slice(start, end)]
        _x = np.arange(start, end)

        # if dims != 1, then we're dealing with MIMO
        if dims != 1:
            _out = np.flip(unpack_array(_out, n=dims), axis=1)
            _in = np.flip(unpack_array(_in, n=dims), axis=1)

        f, axes = plt.subplots(
            dims,
            1,
            figsize=(self._width, 1 + dims * SUBPLOT_HEIGHT),
            dpi=self._screen_dpi,
            sharey=True,
            sharex=True,
        )

        if dims == 1:
            axes = np.array([axes])
            _out = _out.reshape(-1, 1)
            _in = _in.reshape(-1, 1)

        def minmax(x):
            return (x.min(), x.max())

        for i, axis in enumerate(axes):
            _xlim, _ylim = minmax(np.hstack([_out, _in]))
            _lower_margin, _h = 0.15, 0.35

            axis.plot(_x, _out[:, i], marker="+", color="C0", drawstyle="steps-post")
            axis.set_xlim(_x.min() - 0.5, _x.max() + 0.5)
            axis.set_ylim(_out.min() - _lower_margin, _out.max() + _h)
            axis.set_ylabel("Output[{}]".format(i))
            axis.yaxis.label.set_color("C0")
            axis.tick_params(axis="y", color="C0")

            twin = axis.twinx()
            twin.plot(_x, _in[:, i], marker="x", color="C1", drawstyle="steps-post")
            twin.set_ylim(_in.min() - _h, _in.max() + _lower_margin)
            twin.set_ylabel("Input[{}]".format(i))
            twin.yaxis.label.set_color("C1")
            twin.tick_params(axis="y", color="C1")

            twin.spines["left"].set_color("C0")
            twin.spines["right"].set_color("C1")

            axis.grid(axis="y", color="C0", dashes=(5, 5), alpha=0.5)
            twin.grid(axis="y", color="C1", dashes=(5, 5), alpha=0.5)

        axes[-1].xaxis.set_major_locator(plt.MultipleLocator(base=1.0))
        axes[-1].set_xlabel("Symbol #", color="gray")
        if _x.size >= 50:
            plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=90)
        if _x.size >= 60:
            pass

        for axis in axes:
            sns.despine(ax=axis, top=True, bottom=False, right=False, left=False)

        plt.suptitle(
            "Symbol stream for symbols {} to {}".format(*minmax(_x)),
            y=1.01,
            verticalalignment="bottom",
        )
        f.tight_layout()

        self._plot_save_helper(f, **kwargs)

    def plot_eye_diagram(self, **kwargs):
        SUBPLOT_HEIGHT = self._width // 3  # inches

        f, axis = plt.subplots(
            1, 1, figsize=(self._width, SUBPLOT_HEIGHT), dpi=self._screen_dpi
        )

        data: np.ndarray

        # Handle 2-d and 3-d data
        if self.run.i_lnestream.ndim == 2:
            data = self.run.i_lnestream.T[:, : self.run.i_symstream.size]

        elif self.run.i_lnestream.ndim == 3:
            data = np.vstack(self.run.i_lnestream).T[:, : self.run.i_symstream.size]

        axis.plot(data, color="C0", linestyle="-", marker=".", alpha=(10 / data.shape[1]))
        sns.despine()

        lower_ylim, upper_ylim = np.quantile(data, [0.05, 0.95])
        axis.set_ylim(0.975 * lower_ylim, 1.025 * upper_ylim)

        axis.set_xlabel("Sample #", color="gray")
        ylabel = self.run.i_rdpstream.columns[1].split(":")
        ylabel = map(ylabel.__getitem__, [0, 1, -1])
        ylabel = "\n".join(ylabel)
        axis.set_ylabel(ylabel, color="gray")

        plt.suptitle(
            "Eye diagram for a total of {} symbols".format(data.shape[1]),
            verticalalignment="bottom",
        )
        f.tight_layout()

        self._plot_save_helper(f, **kwargs)

    def plot_symbol_space(self, **kwargs):
        symbol_space = self.run.intermediates.lne.symbol_space
        decision_device = self.run.intermediates.lne.decision_device["decision_device"][0]

        if not is_fitted(decision_device):
            raise RuntimeError("decision device must be fitted for symbol space plotting")

        sca = decision_device.named_steps.get("standardscaler", None)
        pca = decision_device.named_steps.get("pca", None)
        cla = decision_device.steps[-1][1]

        X = sca.transform(symbol_space) if sca else symbol_space
        X = pca.transform(symbol_space) if pca else symbol_space

        if X.shape[1] == 1:
            # plotting_data
            f, axes = plt.subplots(
                4,
                1,
                figsize=(self._width, 8),
                dpi=self._screen_dpi,
                sharex=True,
                gridspec_kw={"height_ratios": [2, 3, 2, 1]},
            )

            X = X.ravel()
            pred = self.run.i_bitstream.ravel()
            gt = self.run.o_bitstream.ravel()

            if X.size == pred.size:
                # Jitter the symbol space slightly to avoid covariance calculation errors
                # when all data points are the same. Also, jitter more heavily for the swarm/scatter
                # plot representation to improve readability.
                plotting_data = pandas.DataFrame(
                    {
                        "X": X + 1e-6 * np.random.randn(*X.shape),
                        "Jittered": pred + 0.1 * np.random.randn(*pred.shape),
                        "Prediction": pred,
                        "Error": pred[: gt.size] != gt[: pred.size],
                    }
                )

                sns.scatterplot(
                    x="X",
                    y="Jittered",
                    hue="Prediction",
                    style="Error",
                    legend="brief",
                    alpha=0.3,
                    style_order=[False, True],
                    palette=sns.color_palette("pastel", n_colors=2),
                    data=plotting_data.query("Error == False"),
                    ax=axes[1],
                )

                if plotting_data.query("Error == True").size > 0:
                    n_colors = plotting_data.query("Error == True")["Prediction"].unique().size

                    sns.scatterplot(
                        x="X",
                        y="Jittered",
                        hue="Prediction",
                        style="Error",
                        legend=None,
                        palette=sns.color_palette(palette=None, n_colors=n_colors),
                        style_order=[False, True],
                        data=plotting_data.query("Error == True"),
                        ax=axes[1],
                    )

                sns.distplot(plotting_data.query("Prediction == 0").X, ax=axes[0], color="C0")
                sns.rugplot(
                    plotting_data.query("Prediction == 0").X, alpha=0.5, ax=axes[0], color="C0"
                )
                sns.distplot(plotting_data.query("Prediction == 1").X, ax=axes[0], color="C1")
                sns.rugplot(
                    plotting_data.query("Prediction == 1").X, alpha=0.5, ax=axes[0], color="C1"
                )

                if plotting_data.query("Error == True").size > 0:
                    # ValueError's can be thrown when only a single error exists
                    try:
                        sns.distplot(
                            plotting_data.query("Prediction == 0").query("Error == True").X,
                            ax=axes[2],
                            color="C0",
                        )
                        rugplot(
                            plotting_data.query("Prediction == 0").query("Error == True").X,
                            alpha=0.5,
                            ax=axes[2],
                            color="C0",
                            top=True,
                        )
                    except ValueError:
                        pass

                    try:
                        sns.distplot(
                            plotting_data.query("Prediction == 1").query("Error == True").X,
                            ax=axes[2],
                            color="C1",
                        )
                        rugplot(
                            plotting_data.query("Prediction == 1").query("Error == True").X,
                            alpha=0.5,
                            ax=axes[2],
                            color="C1",
                            top=True,
                        )
                    except ValueError:
                        pass

                axes[2].set_ylim(*reversed(axes[2].get_ylim()))

                for axis in axes:
                    remove_spine(axis, "right")
                for axis in axes[[1, 2]]:
                    remove_spine(axis, "bottom")
                for axis in axes[[0, 1, -1]]:
                    remove_spine(axis, "top")

                add_spine(axes[3], "bottom", ticks_only=True)
                axes[0].grid(dashes=(5, 5), alpha=0.5, axis="x")
                axes[1].grid(dashes=(5, 5), alpha=0.5, axis="x")
                axes[2].grid(dashes=(5, 5), alpha=0.5, axis="x")

                for axis in axes[:-1]:
                    axis.set_xlabel(None)
                    axis.set_ylabel(None)

                axes[1].yaxis.set_ticks(np.unique(plotting_data.Prediction))

                axes[0].set_ylabel("Measurement\ndistribution", color="gray")
                axes[1].set_ylabel("Predicted\nsymbol", color="gray")
                axes[2].set_ylabel("Error\ndistribution", color="gray")

                f.align_ylabels(axes[:])

            else:
                # No known layer uses different encoding at the moment
                pass

            _x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            _d = (
                decision_device.decision_function(_x)
                if hasattr(decision_device, "decision_function")
                else decision_device.predict_proba(_x)
            )

            axes[-1].plot(_x, _d)
            axes[-1].grid(dashes=(5, 5), alpha=0.5)
            axes[-1].set_xlim(0.975 * X.min(), 1.025 * X.max())

            # Labels
            xlabel = self.run.i_rdpstream.columns[1].split(":")
            xlabel = map(xlabel.__getitem__, [0, 1, -1])
            xlabel = "{}, {} ({})".format(*xlabel)
            axes[-1].set_xlabel(xlabel, color="gray")

            ylabel = (
                "Decision\nfunction"
                if hasattr(decision_device, "decision_function")
                else "Prediction\nprobability"
            )
            axes[-1].set_ylabel(ylabel, color="gray")

        else:
            pred = self.run.i_symstream
            gt = self.run.o_symstream

            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()

            resolution = 0.1
            XX, YY = np.meshgrid(
                np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution)
            )

            plotting_data = pandas.DataFrame(
                {
                    "Re": X[:, 0],
                    "Im": X[:, 1],
                    "Symbol": [f"Symbol {x:2b}" for x in pred],
                    "Error": pred[: gt.size] != gt[: pred.size],
                }
            )

            f, axis = plt.subplots(
                1, 1, figsize=(self._width, self._width), dpi=self._screen_dpi
            )

            axis = sns.scatterplot(
                x="Re", y="Im", hue="Symbol", style="Error", data=plotting_data, ax=axis
            )

            axis.set_ylabel("Quadrature")
            axis.set_xlabel("In-phase")
            axis.set_aspect("equal", "box")
            axis.grid(dashes=(5, 5), alpha=0.5)

            try:
                params = AttributeDict()

                if hasattr(cla, "decision_function"):
                    Z = cla.decision_function(np.c_[XX.ravel(), YY.ravel()])
                    step = 1
                    params.levels = np.arange(-1, 1 + step, step)
                    params.linewidths = 1.5 - np.abs(params.levels)
                else:
                    Z = cla.predict_proba(np.c_[XX.ravel(), YY.ravel()])
                    step = 0.5
                    params.levels = np.arange(-1, 1 + step, step)
                    params.linewidths = 1.5 - np.abs(params.levels)

                for dim in range(Z.shape[1]):
                    ZZ = Z[:, dim].reshape(XX.shape)
                    contours = plt.contour(
                        XX,
                        YY,
                        ZZ,
                        colors=[sns.color_palette()[dim]],
                        linestyles=["--", ":"][dim % 2],
                        **params,
                    )
                    plt.gca().clabel(contours, inline=1, fontsize=10)
            except Exception:
                pass

        plt.suptitle("Symbol space", y=0.95, verticalalignment="bottom")
        self._plot_save_helper(f, **kwargs)

    def plot_error(self, roll: t.Optional[int] = None, **kwargs):
        SUBPLOT_HEIGHT = 2

        f, axes = plt.subplots(
            2,
            1,
            figsize=(self._width, 1 + 2 * SUBPLOT_HEIGHT),
            dpi=self._screen_dpi,
            sharey=False,
            sharex=False,
        )

        bit_mismatch_length = self.run.o_bitstream.size - self.run.i_bitstream.size
        sym_mismatch_length = self.run.o_symstream.size - self.run.i_symstream.size

        bit_errors = (
            self.run.i_bitstream[: self.run.o_bitstream.size]
            != self.run.o_bitstream[: self.run.i_bitstream.size]
        )
        sym_errors = (
            self.run.i_symstream[: self.run.o_symstream.size]
            != self.run.o_symstream[: self.run.i_symstream.size]
        )

        bit_x = np.arange(0, bit_errors.size)
        sym_x = np.arange(0, sym_errors.size)

        bit_mismatch = (
            np.arange(bit_errors.size, bit_errors.size + bit_mismatch_length)
            if bit_mismatch_length != 0
            else None
        )
        sym_mismatch = (
            np.arange(sym_errors.size, sym_errors.size + sym_mismatch_length)
            if sym_mismatch_length != 0
            else None
        )

        bit_roll = roll if roll else bit_errors.size // 10
        sym_roll = roll if roll else sym_errors.size // 10

        bit_errors_series = (
            pandas.Series(bit_errors)
            .rolling(window=bit_roll, min_periods=1, center=True)
            .mean()
        )
        sym_errors_series = (
            pandas.Series(sym_errors)
            .rolling(window=sym_roll, min_periods=1, center=True)
            .mean()
        )

        axes[1].plot(bit_x, bit_errors_series)
        axes[0].plot(sym_x, sym_errors_series)

        if bit_mismatch:
            axes[1].plot(bit_mismatch, [1.0] * bit_mismatch, linestyle="--")
        if sym_mismatch:
            axes[1].plot(sym_mismatch, [1.0] * sym_mismatch, linestyle="--")

        axes[1].set_ylim(0, 0.5)
        axes[0].set_ylim(0, 1)

        axes[1].set_xlim(0, bit_errors.size - 1 + bit_mismatch_length)
        axes[0].set_xlim(0, sym_errors.size - 1 + sym_mismatch_length)

        axes[1].set_title(
            "Windowed bit error rate (window={})".format(bit_roll),
            fontstyle="italic",
            color="gray",
        )
        axes[1].set_xlabel("Bit #", color="gray")
        axes[1].set_ylabel("Bit error rate", color="gray")
        axes[0].set_title(
            "Windowed symbol error rate (window={})".format(sym_roll),
            fontstyle="italic",
            color="gray",
        )
        axes[0].set_xlabel("Symbol #", color="gray")
        axes[0].set_ylabel("Symbol error rate", color="gray")

        plt.suptitle("Error rates", verticalalignment="bottom")
        f.tight_layout()

        self._plot_save_helper(f, **kwargs)

    def plot_timing(self, **kwargs):
        SUBPLOT_HEIGHT = 2

        f, axes = plt.subplots(
            2,
            2,
            figsize=(self._width, 1 * 2 * SUBPLOT_HEIGHT),
            dpi=self._screen_dpi,
            sharey="row",
            sharex=False,
            gridspec_kw={"height_ratios": [1, 2]},
        )

        raw_timing = self.run.i_rdpstream.iloc[:, 0]
        raw_timing_delta = raw_timing.diff()

        if self.run.intermediates.rdp.timestamps.ndim < 3:
            rdp_timing = pandas.Series(np.hstack(self.run.intermediates.rdp.timestamps))
        else:
            rdp_timing = pandas.Series(
                np.hstack(self.run.intermediates.rdp.timestamps[:, 0, :])
            )

        rdp_timing_delta = rdp_timing.diff()

        axes[0, 0].plot(np.linspace(0, 1, raw_timing.size), raw_timing)
        axes[0, 1].plot(np.linspace(0, 1, rdp_timing.size), rdp_timing)

        axes[1, 0].plot(np.linspace(0, 1, raw_timing_delta.size), raw_timing_delta)
        axes[1, 1].plot(np.linspace(0, 1, rdp_timing_delta.size), rdp_timing_delta)

        for axis in axes.ravel():
            axis.set_xticks([])

        axes[0, 0].set_ylabel("timestamp (s)", color="gray")
        axes[1, 0].set_ylabel("timestamp diff (s)", color="gray")

        axes[0, 0].set_title(
            "Sample-wise timestamp differences\nRaw data", fontstyle="italic", color="gray"
        )
        axes[0, 1].set_title(
            "Sample-wise timestamp differences\nInterpolated data",
            fontstyle="italic",
            color="gray",
        )

        sns.despine()
        f.tight_layout()

        self._plot_save_helper(f, **kwargs)

    def plot_synchronisation(self, **kwargs):
        SUBPLOT_HEIGHT = 2  # inches

        timestamps = self.run.i_rdpstream.iloc[:, 0]
        data = self.run.i_rdpstream.iloc[:, 1:]
        dims = data.shape[1]

        start = timestamps.iloc[0]
        end = self.run.intermediates.rdp.slicing[10]
        interval = timestamps.between(start, end)

        ZOOM_BEFORE = 3
        ZOOM_AFTER = 5

        origin, *edges = self.run.intermediates.rdp.edge_detection
        slicing = self.run.intermediates.rdp.slicing
        zoom_start = slicing[0] - ZOOM_BEFORE * (slicing[1] - slicing[0])
        zoom_end = slicing[0] + ZOOM_AFTER * (slicing[1] - slicing[0])
        zoom_interval = timestamps.between(zoom_start, zoom_end)

        # Create subplots: 3 columns, ndim rows
        f, axes = plt.subplots(
            dims,
            2,
            figsize=(self._width, 1 + dims * SUBPLOT_HEIGHT),
            dpi=self._screen_dpi,
            sharex="col",
            sharey="row",
            squeeze=False,
            gridspec_kw={"width_ratios": [3, 2]},
        )

        lower_ylim, upper_ylim = np.quantile(data, [0.01, 0.99])

        for i, axis in enumerate(axes[:, 0]):
            axis.plot(
                timestamps[interval],
                data[interval].iloc[:, i],
                marker="+",
                markersize=2,
                linewidth=0.5,
                alpha=0.5,
            )
            axis.set_ylim(0.975 * lower_ylim, 1.025 * upper_ylim)
            axis.set_ylabel(
                "{}\n{}\n{} ({})".format(*data[interval].columns[i].split(":")), color="gray"
            )

            axis.vlines(
                origin,
                0,
                1,
                transform=axis.get_xaxis_transform(),
                linewidth=1.5,
                alpha=0.7,
                linestyle="--",
                color="k",
            )

        axes[0, 0].vlines(
            edges,
            0,
            1,
            transform=axes[0, 0].get_xaxis_transform(),
            linewidth=1.5,
            alpha=0.7,
            linestyle="--",
            color="C1",
        )

        for i, axis in enumerate(axes[:, 1]):
            axis.plot(
                timestamps[zoom_interval],
                data[zoom_interval].iloc[:, i],
                marker="+",
                markersize=2,
                linewidth=0.5,
                alpha=0.5,
            )
            axis.tick_params(axis="x", rotation=45)

            axis.vlines(
                slicing[0 : ZOOM_AFTER + 1],
                0,
                1,
                transform=axis.get_xaxis_transform(),
                linewidth=1.0,
                alpha=0.7,
                linestyle=":",
                color="C0",
            )

            axis.vlines(
                origin,
                0,
                1,
                transform=axis.get_xaxis_transform(),
                linewidth=1.0,
                alpha=0.7,
                linestyle="--",
                color="k",
            )

        sns.despine()

        axes[-1, 0].set_xlabel(self.run.i_rdpstream.columns[0], color="gray")
        axes[-1, 1].set_xlabel(self.run.i_rdpstream.columns[0], color="gray")
        f.align_xlabels(axes[-1, :])

        fmt = lambda x: np.format_float_scientific(x, precision=3)
        axes[0, 0].set_title(
            "Preprocessed data\nin interval {} to {}".format(*map(fmt, [start, end])),
            fontstyle="italic",
            color="gray",
        )
        axes[0, 1].set_title(
            "Preprocessed data\nin interval {} to {}".format(*map(fmt, [zoom_start, zoom_end])),
            fontstyle="italic",
            color="gray",
        )

        plt.suptitle("Synchronisation", y=1.01, verticalalignment="bottom")
        f.tight_layout()
        plt.subplots_adjust(hspace=0.5)

        self._plot_save_helper(f, **kwargs)
