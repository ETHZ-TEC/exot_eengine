from exot.util.factory import GenericFactory

from ._driver import Driver

__all__ = ("DriverFactory",)


class DriverFactory(GenericFactory, klass=Driver):
    pass
