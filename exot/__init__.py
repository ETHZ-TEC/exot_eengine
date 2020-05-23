from .__version__ import __version__
import importlib


if importlib.util.find_spec("tensorflow") is None:
    print("Tensorflow not available - excluding packets using it!")

__all__ = ("__version__",)
