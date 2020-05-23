"""Concrete Android drivers"""

from ._driver import Driver
from ._mixins import ExtraFileMethodsMixin
from .adb import ADBBackend

__all__ = ("ADBAndroidDriver",)


class ADBAndroidDriver(Driver, ExtraFileMethodsMixin, backend=ADBBackend):
    pass
