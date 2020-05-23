# Copyright (c) 2015-2020, Swiss Federal Institute of Technology (ETH Zurich)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
"""Generic type/object factory for class hierarchies"""

import typing as t
from difflib import SequenceMatcher
from operator import itemgetter

from exot.exceptions import AbstractMatchError, AmbiguousMatchError, NothingMatchedError
from exot.util.misc import get_concrete_subclasses, get_subclasses, has_type

__all__ = ("GenericFactory",)


"""
GenericFactory
--------------

The generic factory can be used to generate types using a class hierarchy. For example,
a type that has subclasses (obtained with :meth:`get_concrete_subclasses`) can be
readily used with the factory.

The matching is based on class qualified names, in its current form the package
information is lost. Therefore if the base class `package.base` has derived classes
`package.a.derived` and `package.b.derived`, the factory will raise an ambiguous match
error.
"""


class GenericFactory:
    """Generic object factory

    ..note::
        The type of objects to be used needs to be provided as a 'klass'
        argument, for example:

        >>> class MyFactory(GenericFactory, klass=MyType): pass
    """

    def __init__(self, similarity: float = 1.0):
        """Initialise the factory

        :param similarity: minimum match similarity, (0.0, 1.0],
        defaults to 0.9
        :type similarity: float, optional
        """
        self._available = get_concrete_subclasses(self.klass)
        self._unavailable = [k for k in get_subclasses(self.klass) if k not in self._available]

        if not isinstance(similarity, float):
            raise TypeError("'similarity' should be a float", similarity)

        if not similarity > 0.0 and similarity <= 1.0:
            raise ValueError("'similarity' should be in range (0.0, 1.0]", similarity)

        self._min_similarity = similarity

    def __init_subclass__(cls, klass: type, **kwargs):
        if not isinstance(klass, type):
            raise TypeError(klass)

        if has_type(klass):
            cls._type = klass.Type
            cls._types = [k for k in klass.Type]

        cls._klass = klass
        super().__init_subclass__(**kwargs)

    @property
    def klass(self) -> type:
        return self._klass if hasattr(self, "_klass") else NotImplemented

    @property
    def available(self) -> t.Set[type]:
        return self._available

    @property
    def default(self) -> type:
        return self.available[0] if self.available else None

    @property
    def unavailable(self) -> t.Set[type]:
        return self._unavailable

    @property
    def available_types(self) -> t.Optional[t.List]:
        return self._types if hasattr(self, "_types") else None

    def update_registry(self):
        self._available = get_concrete_subclasses(self.klass)
        self._unavailable = [k for k in get_subclasses(self.klass) if k not in self._available]

    def match(self, name: str, variant: str = "concrete") -> t.List[t.Dict]:
        if not isinstance(variant, str):
            raise TypeError("expected a str for 'variant'", variant)

        allowed = ["concrete", "abstract", "all"]
        if variant not in allowed:
            raise ValueError(f"'variant' has to be one of {allowed!r}", variant)

        if variant == "concrete":
            search_space = self.available
        elif variant == "abstract":
            search_space = self.unavailable
        else:
            search_space = self.available + self.unavailable

        sims = [
            {
                "class": k,
                "name": k.__qualname__,
                "similarity": SequenceMatcher(None, k.__qualname__, name).quick_ratio(),
            }
            for k in search_space
        ]

        sims.sort(key=itemgetter("similarity"), reverse=True)

        if len(sims) > 1 and any(m["similarity"] == sims[0]["similarity"] for m in sims[1:]):
            raise AmbiguousMatchError("multiple equal-importance matches", sims)

        return sims

    def produce_type(self, name: str, **kwargs) -> type:
        if "variant" in kwargs:
            variant = kwargs["variant"]
        else:
            variant = "concrete"

        matches = self.match(name, variant)

        if not matches:
            raise NothingMatchedError(f"request '{name}' not found")

        top_match = matches[0]

        if top_match["similarity"] < self._min_similarity:
            if variant == "concrete":
                abstract = self.match(name, variant="abstract")
                if abstract and abstract[0]["similarity"] >= self._min_similarity:
                    raise AbstractMatchError(
                        f"request '{name}' found in abstract classes: {abstract[:2]!r}"
                    )

            raise ValueError(
                "request '{n}' matched with insufficient sim. ({s:.2f} < {m:.2f}); "
                "best match: {match!r}".format(
                    n=name, s=top_match["similarity"], m=self._min_similarity, match=top_match
                )
            )

        return top_match["class"]

    def verify_type(self, klass: type, **kwargs) -> None:
        """Verify if a type matches the desired type"""
        if "_type" in kwargs and hasattr(self, "_type"):
            _type = kwargs.pop("_type")
            _klass_type = getattr(klass, "__type__", None)
            if _klass_type is not _type:
                raise ValueError(
                    f"match '{klass!s}' is of different type ('{_klass_type}') than "
                    f"requested ({_type})"
                )

    def __call__(self, name: str, *args, **kwargs) -> object:
        klass = self.produce_type(name)
        self.verify_type(klass, **kwargs)
        kwargs.pop("_type", None)
        return klass(*args, **kwargs)

    def make_default(self, *args, **kwargs) -> object:
        klass = self.default
        return klass(*args, **kwargs)
