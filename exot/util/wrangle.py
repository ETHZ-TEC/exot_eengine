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
"""Utilities for experiment handling, data-wrangling, parsing, etc."""

from __future__ import annotations

import functools
import operator
import typing as t
from pathlib import Path
from string import whitespace

import numpy as np
import pandas as pd

from exot.exceptions import MatchNotFound

__all__ = (
    "app_log_formatter",
    "log_path_unformatter",
    "repetition_formatter",
    "app_configfile_formatter",
    "CATEGORIES",
    "filter_data",
    "filter_factory",
    "get_unique_column_combinations",
    "make_data_filter_array",
    "map_factory",
    "Matcher",
    "parse_and_describe",
    "parse_header",
    "parse_header_entry",
    "run_path_formatter",
    "run_path_unformatter",
)


CATEGORIES = ["module", "value", "dimension", "unit", "quantity", "method"]


def map_factory(where, which):
    return map(lambda x: x[which], where)


def filter_factory(where, which, value):
    return filter(lambda x: x[which] == value, where)


def run_path_formatter(*param: t.Hashable) -> str:
    """Produce a formatted path from parameters

    Args:
        param (t.Hashable): a sequence of hashable params

    ..note::
        The function aims to proide a unified way of translating between parameters
        in phases and strings

        Since only hashable types can be used as dict keys, only hashable types
        are allowed in the run path. Strings should be stripped. If resulting path
        has whitespace in it, an exception will be thrown.

        Int values are fixed to 5 digits, with leading zeros.

        While 'phases' can have have different depths, we will tentatively constrain
        the file hierarchy depth to 1. For example: these are valid paths:

        >>> run_path_formatter("run", 1) # run_00001
        >>> run_path_formatter("eval", 123) # eval_00123
        >>> run_path_formatter("run", "param1", 1) # run_param1_00001

        These are some of the invalid paths:

        >>> run_path_formatter("train", ("float", 0.1))
        >>> run_path_formatter("train", {1: 1})
        >>> run_path_formatter("train", [1, 2])
    """

    def visitor(x):
        if not isinstance(x, t.Hashable):
            raise TypeError("like dict keys, a run-path cannot have non-hashable components", x)

        if isinstance(x, (float, np.floating)):
            raise TypeError("float values are not allowed in a run path", x)

        if isinstance(x, (int, np.integer)):
            return "{:>06d}".format(x)
        elif isinstance(x, str):
            assert (
                "_" not in x
            ), f"run_path_formatte uses '_' as a special separator, {x} is invalid"
            return "{}".format(x)
        elif isinstance(x, t.Tuple):
            return "{}".format("-".join(visitor(a) for a in x))
        else:
            return "{}".format(x)

    formatted = "/".join(visitor(x) for x in param)

    if any(c in whitespace for c in formatted):
        raise ValueError(f"formatted path contained whitespace: {formatted!r}")

    return formatted


def run_path_unformatter(path: Path) -> t.Tuple:
    """Produce parameters from a formatted path"""

    assert isinstance(path, Path), "path must be a str"
    assert path.name == "_run.pickle", "path does not point to a _run.pickle file"
    split = [path.parent.parent.name, path.parent.name]

    def visitor(x):
        assert not any(c in whitespace for c in x), "no whitespace in valid path"
        if "-" in x:
            return tuple(visitor(_x) for _x in x.split("-"))
        elif x.isdigit():
            return int(x)
        else:
            return x

    return tuple(visitor(x) for x in split)


def repetition_formatter(rep: int) -> Path:
    return Path("{:>03d}".format(rep))


def app_configfile_formatter(app: str) -> Path:
    """Produce a formatted json filename"""

    return "{}.config.json".format(app)


def app_log_formatter(app: str, dbg: bool) -> Path:
    """Produce a formatted log filename"""

    return "{}.{}".format(app, "debug.txt" if dbg else "log.csv")


def generic_log_formatter(app: str, dbg: bool) -> Path:
    """Produce a formatted log filename"""

    return "{}.{}".format(app, "stderr.txt" if dbg else "stdout.txt")


def log_path_unformatter(path: Path) -> (int, str, bool):
    """Produce a formatted log filename"""

    _split_by_dot = path.name.split(".")
    assert len(_split_by_dot) == 3, "expect 3 comma-separated components"

    env = path.parent.parent.name
    rep = int(path.parent.name)
    app = _split_by_dot[0]
    dbg = True if _split_by_dot[1] == "debug" else False

    return env, rep, app, dbg


def parse_header_entry(entry: str) -> dict:
    """Parses a header entry

    Args:
        entry (str): The header entry

    Returns:
        dict: The parsed entry (dict with keys: module, value, dimension, unit)
    """

    assert isinstance(entry, str), "entry must be a string"
    _categories = ["module", "value", "dimension", "unit"]
    _split_by_colon = entry.split(":")
    assert len(_split_by_colon) == 4, f"expected 4 colon-separated elements in entry"
    _dict = dict(zip(_categories, _split_by_colon))
    _dict["quantity"] = _dict["module"].split("_")[0] if "_" in _dict["module"] else ""
    _dict["method"] = _dict["module"].split("_")[1] if "_" in _dict["module"] else ""
    return _dict


def parse_header(header: t.Union[t.Iterable, str]) -> t.List[dict]:
    """Parses the entire header

    Args:
        header (t.Union[t.Iterable, str]): The header

    Returns:
        t.List[dict]: A list of parsed header entries
    """

    assert isinstance(header, (t.Iterable, str)), "must be a str or a list"
    header = header.split(",") if isinstance(header, str) else header
    return list(map(parse_header_entry, header))


def parse_and_describe(header: t.Union[t.Iterable, str]) -> dict:
    """Parses and describes the header

    Args:
        header (t.Union[t.Iterable, str]): The header

    Returns:
        dict: A dict with available modules, quantities, and methods
    """

    _parsed = parse_header(header)
    _modules = {_["module"] for _ in _parsed}
    _quantities = {_["quantity"] for _ in _parsed if _["quantity"]}
    _methods = {_["method"] for _ in _parsed if _["method"]}
    return {
        "available_modules": _modules,
        "available_quantities": _quantities,
        "available_methods": _methods,
    }


def make_data_filter_array(
    data: t.Union[pd.DataFrame, pd.Series], op=operator.eq, **queries: t.Any
) -> pd.Series:
    """Make an array to filter Pandas data using logical and operation

    Args:
        data (t.Union[pd.DataFrame, pd.Series]): The data to filter
        **queries (t.Any): The queries

    Returns:
        pd.Series: The boolean array for row selection

    Raises:
        TypeError: Wrong type supplied
        ValueError: Unavailable key specified
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError(
            "'make_data_filter_array' can only filter Pandas types, got: {type(data)}."
        )

    if not queries:
        _ = pd.Series(np.ones_like(data.index).astype(bool))
        _.index - data.index
        return _

    unavailable = [key for key in queries if key not in data]
    if unavailable:
        raise ValueError(f"Data selection failed, unavailable keys requested: {unavailable}")

    return functools.reduce(
        np.logical_and,
        [
            op(data[key], value)
            if not isinstance(value, (t.List, np.ndarray))
            else np.isin(data[key], value)
            for key, value in queries.items()
        ],
    )


def filter_data(
    data: t.Union[pd.DataFrame, pd.Series], **queries: t.Any
) -> t.Union[pd.DataFrame, pd.Series]:
    """Filters Pandas data using queries

    Args:
        data (t.Union[pd.DataFrame, pd.Series]): The data
        **queries (t.Any): The queries

    Returns:
        t.Union[pd.DataFrame, pd.Series]: The filtered data frame

    Raises:
        TypeError: Wrong type supplied to 'data'
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError(
            f"'filter' can only filter Pandas data types, got: {type(data)}; "
            "use regular indexing for NumPy data types."
        )

    return data[make_data_filter_array(data, **queries)]


def filter_with_callable(data: t.T, callable: t.Callable) -> t.T:
    """Filters data with a callable

    Args:
        data (t.T): The data
        callable (t.Callable): The callable

    Returns:
        t.T: The filtered data
    """
    return type(data)(filter(callable, data))


def get_unique_column_combinations(data: pd.DataFrame, columns=[]) -> t.Tuple:
    """Gets the unique combinations of columns

    Args:
        data (pd.DataFrame): The data frame
        columns (list, optional): The columns to choose. Defaults to [].

    Raises:
        TypeError: Wrong type provided to 'data'

    Returns:
        t.Tuple: The unique combinations of columns
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("'get_unique_column_combinations' must receive a Pandas DataFrame")

    return data[columns].drop_duplicates().apply(tuple, axis=1).reset_index(drop=True)


class Matcher:

    """The Matcher can be used in DataFrame selection, thanks to its `__call__` method.

    For details on how to use a Callable for DataFrame selection, see:
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#selection-by-callable>
    """

    def __init__(self, quantity: str, method: str, values: list, dimensions: list):
        """Instantiate a Matcher

        Args:
            quantity (str): a desired quantity
            method (str): a desired method for the quantity
            values (list): a set of desired values for the quantity and method
            dimensions (list): a set of desired dimensions of the values
        """
        assert isinstance(quantity, str), "'quantity' must be a string"
        assert isinstance(method, str), "'method' must be a string"
        assert isinstance(values, t.List), "'values' must be a list"
        assert isinstance(dimensions, t.List), "'dimensions' must be a list"

        self._quantity = quantity
        self._method = method
        self._values = values
        self._dimensions = list(map(str, dimensions))
        self._df_columns = None

    def __repr__(self) -> str:
        """Represent the Matcher object

        Returns:
            str: a repr-like string representation
        """
        _ = "<Matcher quantity={!r}, method={!r}, values={!r}, dimensions={!r}>"
        return _.format(self._quantity, self._method, self._values, self._dimensions)

    @classmethod
    def from_dict(cls, src: dict) -> Matcher:
        """Instantiates a Matcher from a dictionary

        Args:
            src (dict): The dictionary

        Returns:
            Matcher: The Matcher instance
        """
        return cls(
            src.get("quantity"), src.get("method"), src.get("values"), src.get("dimensions")
        )

    def validate(self, df: t.Union[pd.DataFrame, pd.Series, pd.Panel]) -> t.NoReturn:
        """Check if the provided parameters can be found in the input DataFrame

        Args:
            df (t.Union[pd.DataFrame, pd.Series, pd.Panel]): the DataFrame

        Raises:
            MatchNotFound: If a match could not be found
        """
        parsed = parse_header(df.columns)

        assert len(df.columns) == len(parsed)

        try:
            assert self._quantity in map_factory(parsed, "quantity"), self._quantity
            quantity = list(filter_factory(parsed, "quantity", self._quantity))
            assert self._method in map_factory(quantity, "method"), self._method
            method = list(filter_factory(quantity, "method", self._method))
            assert all(_ in map_factory(method, "value") for _ in self._values), self._values

            for value in self._values:
                __ = list(filter_factory(method, "value", value))
                assert all(
                    _ in map_factory(__, "dimension") for _ in self._dimensions
                ), self._dimensions
        except AssertionError as e:
            raise MatchNotFound(repr(self), df.columns, e.args)

    def matches(self, entry: str) -> bool:
        """Check if a column matches the parameters

        Args:
            entry (str): a column name

        Returns:
            bool: True if matches, False otherwise
        """
        parsed = parse_header_entry(entry)

        if all(
            [
                parsed["quantity"] == self._quantity,
                parsed["method"] == self._method,
                parsed["value"] in self._values,
                parsed["dimension"] in self._dimensions,
            ]
        ):
            return True
        else:
            return False

    def is_timestamp(self, entry: str) -> bool:
        """Check if a column is a timestamp column

        Args:
            entry (str): a column name

        Returns:
            bool: True if contains a timestamp, False otherwise
        """
        parsed = parse_header_entry(entry)
        if (
            parsed["module"] in ["host", "meter", "generator"]
            and parsed["value"] == "timestamp"
        ):
            return True
        else:
            return False

    def get_timestamp_column(self, columns: list) -> str:
        """Get the timestamp column

        Args:
            columns (list): DataFrame columns

        Returns:
            str: The timestamp column

        Raises:
            ValueError: If the DataFrame doesn't have a timestamp column
        """
        _ = [_ for _ in columns if self.is_timestamp(_)]
        if _:
            return _[0]
        else:
            raise ValueError("the provided DataFrame does not have a timestamp column")

    def __call__(self, df: t.Union[pd.DataFrame, pd.Series, pd.Panel]) -> t.List[str]:
        """Return a list of column names that match the parameters

        Args:
            df (t.Union[pd.DataFrame, pd.Series, pd.Panel]): the DataFrame

        Returns:
            t.List[str]: a list of column names
        """
        self.validate(df)
        return [
            self.get_timestamp_column(df.columns),
            *[column for column in df.columns if self.matches(column)],
        ]
