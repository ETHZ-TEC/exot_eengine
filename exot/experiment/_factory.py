from exot.channel import ChannelFactory
from exot.driver import DriverFactory
from exot.util.factory import GenericFactory

from ._base import Experiment

__all__ = "ExperimentFactory"


class ExperimentFactory(GenericFactory, klass=Experiment):
    def make(self, experiment: str, *args, **kwargs):
        _experiment_type = self.produce_type(experiment, variant="all")
        self.verify_type(_experiment_type, **kwargs)

        if "driver" in kwargs:
            driver = DriverFactory()(kwargs["driver"])
            kwargs["driver"] = driver

        if "channel" in kwargs:
            channel = ChannelFactory()(kwargs["channel"])
            kwargs["channel"] = channel

        return _experiment_type(*args, **kwargs)
