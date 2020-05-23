"""Layer factory"""

from exot.util.factory import GenericFactory

from ._base import Layer


class LayerFactory(GenericFactory, klass=Layer):
    pass
