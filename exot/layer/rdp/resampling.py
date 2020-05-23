"""Resampling RDP layers"""

from .._base import Layer


class Resampling(Layer, layer=Layer.Type.PrePost):
    pass


class Downsampling(Resampling):
    pass


class Upsampling(Resampling):
    pass


class Undersampling(Resampling):
    pass


class Oversampling(Resampling):
    pass
