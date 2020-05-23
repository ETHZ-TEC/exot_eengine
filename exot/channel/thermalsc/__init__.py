"""Special modules used for the thermal side channel"""
import importlib

from .dataaugmentation import DataAugmentation

if importlib.util.find_spec("tensorflow") is not None:
    from .appdetection import AppDetection

__all__ = (
    "DataAugmentation",
    "AppDetection",
)

