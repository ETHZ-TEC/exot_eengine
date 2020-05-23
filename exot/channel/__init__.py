"""Channel base class

Channel-derived classes define channel-specific properties.
"""
from ._base import (
    Channel,
    ChannelFactory,
    Analysis
  )
from .covertchannels import CovertChannel, FrequencyCC, PowerCC, ThermalCC, CacheCC
from .sidechannels import SideChannel, ThermalSC

__all__ = (
    "Channel",
    "CovertChannel",
    "SideChannel",
    "ChannelFactory",
    "FrequencyCC",
    "PowerCC",
    "ThermalCC",
    "ThermalSC",
    "CacheCC",
)

