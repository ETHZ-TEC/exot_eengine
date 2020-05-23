""" TODO
"""
import abc
import numpy as np
from numpy import inf
import typing as t

from ._base import Channel
from .mixins.covertchannel import CapacityDiscrete, CapacityContinuous, PerformanceSweep

class CovertChannel(Channel):
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

"""
Cache Covert Channels
----------

TODO
"""

class CacheCC(CovertChannel, CapacityDiscrete, PerformanceSweep):
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

class FlushFlushCC(CacheCC):
    pass

class FlushReloadCC(CacheCC):
    pass

class FlushPrefetchCC(CacheCC):
    pass

"""
Frequency Covert Channels
----------

TODO
"""

class FrequencyCC(CovertChannel, CapacityDiscrete):
    pass

class ConservativeGovernorCC(FrequencyCC):
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

"""
Power Cache Covert Channels
----------

TODO
"""

class PowerCC(CovertChannel, CapacityContinuous, PerformanceSweep):
    @property
    def inputs(self) -> np.array:
        """
        0  -> Do not utilise core (cool down)
        n  -> Utilise n cores
        """
        return np.arange(999)

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

    @property
    def analyses_classes(self):
        return {}  # TODO

"""
Thermal Covert Channels
----------

TODO
"""

class ThermalCC(CovertChannel, CapacityDiscrete, PerformanceSweep):
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

    @property
    def analyses_classes(self):
        return {}  # TODO

