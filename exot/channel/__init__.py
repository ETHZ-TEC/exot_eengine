"""Channel base class

Channel-derived classes define channel-specific properties.
"""

import abc
import typing as t

import numpy as np
from numpy import inf

from exot.util.factory import GenericFactory
from exot.util.mixins import SubclassTracker

__all__ = (
    "Channel",
    "ChannelFactory",
    "FrequencyGovernorConservative",
    "Thermal",
    "Power",
    "Cache",
    #    "CacheDirect",
)


class Channel(SubclassTracker, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def signal(self) -> t.Mapping:
        pass

    @property
    def max_num_symbols(self) -> int:
        return len([k for k in self.signal if isinstance(k, int)])

    @property
    @abc.abstractmethod
    def fixed_length_symbols(self) -> bool:
        pass

    @property
    def symbols(self) -> t.List:
        return list(self.signal.keys())

    def __repr__(self) -> str:
        return (
            f"<{self.__module__}.{self.__class__.__name__} at {hex(id(self))} "
            f"signal={self.signal}, fixed={self.fixed_length_symbols}>"
        )


class ChannelFactory(GenericFactory, klass=Channel):
    pass


class Power(Channel):
    @property
    def signal(self) -> t.Mapping:
        """
        0 -> Do not utilise core (low power consumption)
        1 -> Utilise core (high power consumption)
        """
        return {0: 0, 1: 1}

    @property
    def fixed_length_symbols(self) -> bool:
        return True


class FrequencyGovernorConservative(Channel):
    @property
    def signal(self) -> t.Mapping:
        """
        -X -> Decrease frequency by X levels (if possible)
         X -> Increase frequency by X levels (if possible)
        """
        return {
            0: [1, -1],
            1: [-1, 1],
            "preamble": [5, -1],
            "postamble": [1, -5],
            inf: [inf, inf],
        }

    @property
    def fixed_length_symbols(self) -> bool:
        return False


class Thermal(Channel):
    @property
    def inputs(self) -> np.array:
        """
        0  -> Do not utilise core (cool down)
        -1 -> Fully utilise all cores (heat up)
        """
        return np.array([0, 1])

    @property
    def signal(self) -> t.Mapping:
        return {0: [0, -1], 1: [-1, 0], "carrier": [1, 0]}

    @property
    def fixed_length_symbols(self) -> bool:
        return True


class Cache(Channel):
    @property
    def signal(self) -> t.Mapping:
        """
        0 -> Keep data in cache (low access time)
        1 -> Remove data from cache (high access time)
        """
        return {0: 0, 1: 1}

    @property
    def fixed_length_symbols(self) -> bool:
        return True


# class CacheDirect(Channel):
#    @property
#    def signal(self) -> t.Mapping:
#        return {}
#
#    @property
#    def fixed_length_symbols(self) -> bool:
#        return True
#
