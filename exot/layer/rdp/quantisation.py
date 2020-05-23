"""Quantising RDP layer"""

from .._base import Layer


class Quantisation(Layer, layer=Layer.Type.PrePost):
    pass
