"""Passthrough line coding"""

from .._base import Layer


class PassthroughLineCoding(Layer, layer=Layer.Type.Line):
    def _encode(self, upper):
        return upper

    def _decode(self, lower):
        return lower

    def symrate_to_subsymrate(self, symrate):
        return symrate
