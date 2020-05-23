"""Logging configuration"""

import abc
import logging
import re
import typing as t
from pathlib import Path

import coloredlogs

from exot.util.file import add_timestamp
from exot.util.misc import has_property

__all__ = (
    "abbreviate",
    "long_log_formatter",
    "configure_root_logger",
    "get_root_logger",
    "get_logger",
    "Loggable",
)

ROOT_LOGGER_NAME = "exot"
DEFAULT_STREAM_HANDLER_LOG_LEVEL = logging.INFO
LONG_LOG_FORMAT_STR = (
    "%(asctime)s :: %(levelname)-8s :: %(threadName)-10s :: %(name)-20s :: "
    "%(filename)10s:%(lineno)03d (%(funcName)s) :: %(message)s"
)
SHORT_LOG_FORMAT_STR = "%(asctime)s :: %(message)s"


# Format example:
# 1970-01-01 00:00:00,000 :: INFO :: ClassName :: file.py:123 (function_name) :: Message
def long_log_formatter() -> logging.Formatter:
    """Get default logging string format"""
    return logging.Formatter(fmt=LONG_LOG_FORMAT_STR)


def short_log_formatter() -> logging.Formatter:
    """Get a more concise logging string format"""
    return logging.Formatter(fmt=SHORT_LOG_FORMAT_STR)


def abbreviate(s: str, threshold: int = 7, limit: int = 3) -> str:
    """Abbreviate a string, taking accound of camel case to preserve some legibility"""
    assert isinstance(s, str)

    if len(s) > threshold:
        camel_split = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")
        splitted = camel_split.findall(s)
        return "".join(_[: min([len(_), limit])] for _ in splitted)

    return s


def configure_root_logger(path: t.Optional[Path] = None, name: t.Optional[str] = None) -> None:
    """Configure the root logger"""
    global ROOT_LOGGER
    global ROOT_LOGGER_NAME

    logging.captureWarnings(True)
    logging.root.disabled = True
    logging.root.setLevel(logging.NOTSET)

    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    ROOT_LOGGER = logging.getLogger(ROOT_LOGGER_NAME)
    ROOT_LOGGER.setLevel(logging.NOTSET)

    if ROOT_LOGGER.handlers:
        for handler in ROOT_LOGGER.handlers:
            ROOT_LOGGER.removeHandler(handler)

    ROOT_LOGGER.addHandler(logging.StreamHandler())
    ROOT_LOGGER.handlers[0].setLevel(logging.CRITICAL)
    ROOT_LOGGER.handlers[0].setFormatter(short_log_formatter())

    if path and path.is_dir():
        name = name if name else "exot"
        filename = Path("{}.log".format(name))
        filename = add_timestamp(filename)
        filepath = Path.joinpath(path, filename)
        file_logger = logging.FileHandler(filepath)
        file_logger.setFormatter(long_log_formatter())
        file_logger.setLevel(logging.NOTSET)
        ROOT_LOGGER.addHandler(file_logger)


def get_root_logger(**kwargs):
    """Get the root logger"""
    if "ROOT_LOGGER" not in globals():
        configure_root_logger(**kwargs)
    return globals()["ROOT_LOGGER"]


def get_logger(obj: t.Union[type, object], *args, **kwargs):
    """Get a logger instance for a type or an object"""
    global ROOT_LOGGER_NAME
    name = obj.__name__ if isinstance(obj, type) else obj.__class__.__name__
    name = abbreviate(name)
    return get_root_logger().getChild(name, *args, **kwargs)


class Loggable(metaclass=abc.ABCMeta):
    """A mixin class for adding a logger"""

    @property
    def logger(self):
        return self._logger

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, value):
        self.__dict__ = value
        self._bootstrap_logger()

    def set_logging_level(self, value: int):
        if value not in [0, 10, 20, 30, 40, 50]:
            raise ValueError("Only valid log levels are accepted, got:", value)

        for handler in [*self.logger.handlers, *self.logger.parent.handlers]:
            handler_name = handler.__class__.__name__
            if "Stream" in handler_name or "StandardError" in handler_name:
                handler.setLevel(value)

    def _bootstrap_logger(self, custom_filename: t.Optional[str] = None) -> None:
        """Bootstrap the logger

        Will perform the following:
        - Create a filename for logging that contains name, type name, and a timestamp
        - Check if a class has a `log_path` property
        - Delete old file-based loggers
        - Add a StreamHandler and FileHandler to the class logger
        """
        self._logger = get_logger(self)
        self._logger.propagate = True

        experiment = self.name if getattr(self, "name", None) else None
        self_name = self.__class__.__name__
        name = experiment if experiment else self_name
        full_name = ".".join([name, custom_filename]) if custom_filename else name

        filename = Path("{}.log".format(full_name))
        filename = add_timestamp(filename)

        log_paths = []

        if has_property(self, "log_path"):
            if self.log_path and not [
                _ for _ in get_root_logger().handlers if isinstance(_, logging.FileHandler)
            ]:
                log_paths.append(self.log_path)

        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)

        # coloredlogs.install adds a stream handler to the provided logger
        coloredlogs.install(
            level=DEFAULT_STREAM_HANDLER_LOG_LEVEL,
            logger=self.logger,
            fmt=SHORT_LOG_FORMAT_STR,
            isatty=True,
        )

        for log_path in log_paths:
            log_path.mkdir(parents=True, exist_ok=True)
            filepath = log_path / filename

            file_logger = logging.FileHandler(filepath)
            file_logger.setFormatter(long_log_formatter())
            file_logger.setLevel(logging.NOTSET)

            self.logger.addHandler(file_logger)

    def __init__(self, *args, **kwargs):
        """Inits the class logger"""
        self._bootstrap_logger()
