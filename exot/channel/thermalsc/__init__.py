"""Special modules used for the thermal side channel"""
import importlib

from .dataaugmentation import DataAugmentation

if importlib.util.find_spec("tensorflow") is not None:
    from .appdetection import AppDetection
else:
    print("Tensorflow not available - excluding packets relying on it!") # TODO make log message

__all__ = (
    "DataAugmentation",
    "AppDetection",
)

