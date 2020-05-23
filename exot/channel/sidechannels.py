""" TODO
"""
import importlib

import numpy as np
from numpy import inf
import typing as t

from exot.util.logging import get_root_logger

from ._base import Channel
from .thermalsc import DataAugmentation

if importlib.util.find_spec("tensorflow") is not None:
    from .thermalsc import AppDetection

class SideChannel(Channel):
    pass


"""
Thermal Side Channels
----------

TODO
"""
class ThermalSC(SideChannel):
    @property
    def analyses_classes(self):
        if importlib.util.find_spec("tensorflow") is not None:
            return {"DataAugmentation":DataAugmentation, "AppDetection":AppDetection}  # TODO this can be done more nicely
        else:
            return {"DataAugmentation":DataAugmentation}  # TODO this can be done more nicely

