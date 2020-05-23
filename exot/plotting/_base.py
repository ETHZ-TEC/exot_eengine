"""Plotting support for experimental runs"""

import contextlib
import inspect
import typing as t

import matplotlib.pyplot as plt

from exot.util.plotting import _save_path_helper


class Plotter(contextlib.ContextDecorator):
    """Plotting support for Run instances

    Attributes:
        PLOT_FILENAMES (dict): Default filenames for supported plots
    """

    PLOT_FILENAMES = {}

    def __init__(self, *args, save_png: bool = False, save_pdf: bool = False, **kwargs):
        self._screen_dpi: int = kwargs.get("screen_dpi", 72)
        self._print_dpi: int = kwargs.get("print_dpi", 300)
        self._save_path = kwargs.get("save_path", "./")
        self._save_png, self._save_pdf = save_png, save_pdf
        self._width: float = kwargs.get("width", 12)
        self._global_add_to_title = kwargs.get("add_to_title", None)
        self._global_add_to_filename = kwargs.get("add_to_filename", None)

    @property
    def save_path(self):
        return getattr(self, "_save_path", "./")

    @save_path.setter
    def save_path(self, value):
        self._save_path = _save_path_helper(value)

    def __repr__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def _plot_save_helper(self, figure: plt.Figure, **kwargs: t.Any):
        # Get the name of the calling function, look up an entry in the PLOT_FILENAMES dictionary,
        # or, if unavailable, set the fileneme to the name of the plotting function.
        caller: str = inspect.stack()[1][3]
        filename: str = (
            self.PLOT_FILENAMES[caller]
            if caller in self.PLOT_FILENAMES
            else caller.split("plot_")[-1]
        )

        add_to_title = None

        if "add_to_title" in kwargs:
            add_to_title = kwargs.get("add_to_title", "")
            if not isinstance(add_to_title, str):
                add_to_title = str(add_to_title)
        elif self._global_add_to_title is not None:
            add_to_title = self._global_add_to_title

        if add_to_title:
            title = "{} | {}".format(figure._suptitle.get_text(), add_to_title)
            figure._suptitle.set_text(title)

        if "add_to_filename" in kwargs:
            filename = "{}{}".format(filename, kwargs.get("add_to_filename"))
        elif self._global_add_to_filename is not None:
            filename = "{}{}".format(filename, self._global_add_to_filename)

        if self._save_png:
            figure.savefig(
                (self.save_path / filename).with_suffix(".png"),
                dpi=self._screen_dpi,
                bbox_inches="tight",
            )
        if self._save_pdf:
            figure.savefig(
                (self.save_path / filename).with_suffix(".pdf"),
                dpi=self._print_dpi,
                bbox_inches="tight",
            )
