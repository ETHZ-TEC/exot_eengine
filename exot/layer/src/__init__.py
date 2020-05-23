"""Source coding layer

Transforms message bits into into symbols and vice versa.
"""

from ._base import SourceCoding
from .bitset import BitsetCoding
from .huffman import Huffman
from .passthrough import SourcePassthrough

__all__ = ("SourceCoding", "Huffman", "SourcePassthrough", "BitsetCoding")
