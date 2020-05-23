"""TODO
"""
import abc
import logging
import typing as t

import numpy as np
from numpy import inf

from exot.util.attributedict import AttributeDict
from exot.util.logging import get_root_logger, Loggable, long_log_formatter
from exot.util.factory import GenericFactory
from exot.util.file import (
    move_action,
)
from exot.util.mixins import (
    SubclassTracker,
)

class Channel(SubclassTracker, metaclass=abc.ABCMeta):
    # Due to cyclic imports the HasParent class cannot be used here. This might need to be fixed in a general revision of ExOT.
    @property
    def parent(self) -> object:
        return getattr(self, "_parent", None)

    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def hasparent(self):
        return self.parent is not None

    @property
    def analyses(self):
        if not hasattr(self, '_analyses'):
            self._analyses = dict()
        return self._analyses

    @property
    @abc.abstractmethod
    def analyses_classes(self):
        pass

    def bootstrap_analyses(self):
        if "ANALYSES" in self.parent.config:
            for analysis in self.parent.config.ANALYSES:
                self.bootstrap_analysis(analysis)
        else:
            get_root_logger().info("No analyses specified")

    def bootstrap_analysis(self, analysis_name: str):
        if "ANALYSES" in self.parent.config:
            if analysis_name in self.parent.config.ANALYSES:
                kwargs        = self.parent.config.ANALYSES[analysis_name].copy()
                kwargs.update(name=analysis_name)
                args          = [self.parent]
                if "type" in kwargs:
                    if kwargs["type"] in self.analyses_classes:
                        new_analysis = self.analyses_classes[kwargs["type"]](*args, **kwargs)
                        self.analyses[new_analysis.name] = new_analysis
                    else:
                        get_root_logger().critical(f"Analysis of type %s not available for channel {type(self)}" % (kwargs["type"]))
                else:
                    get_root_logger().critical(f"type not specified in config for analysis {analysis_name}")
        else:
            get_root_logger().critical(f"Analysis {analysis_name} not specified in configuration!")

class ChannelFactory(GenericFactory, klass=Channel):
    pass

class Analysis(SubclassTracker, metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        self.config = AttributeDict(kwargs)
        self.experiment = args[0]

    def execute(self, *args, **kwargs):
        if self.path.exists():
            self.experiment.logger.info(move_action(self.path))
        self.path.mkdir(exist_ok=True)

        # Make sure the logfiles are also written to the analysis directory
        file_logger = logging.FileHandler(self.log_path)
        file_logger.setFormatter(long_log_formatter())
        file_logger.setLevel(logging.NOTSET)
        self.experiment.logger.addHandler(file_logger)

        self._execute_handler(self, *args, **kwargs)

    @abc.abstractmethod
    def _execute_handler(self, *args, **kwargs):
        pass

    @property
    def name(self):
        return self.config.name

    @property
    @abc.abstractmethod
    def path(self):
        pass

    @property
    def log_path(self):
        return self.path.joinpath("debug.log")

