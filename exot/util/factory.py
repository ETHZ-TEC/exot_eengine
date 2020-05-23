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
        """Gets the base class "produced" by the factory

        Returns:
            type: The factory's base class
        """
        return self._klass if hasattr(self, "_klass") else NotImplemented

    @property
    def available(self) -> t.Set[type]:
        """Gets the available classes that can be created

        Returns:
            t.Set[type]: The set of available classes
        """
        return self._available

    @property
    def default(self) -> type:
        """Gets the default production class

        Returns:
            type: The default production class
        """
        return self.available[0] if self.available else None

    @property
    def unavailable(self) -> t.Set[type]:
        """Gets the unavailable classes that cannot be created (e.g. abstract classes)

        Returns:
            t.Set[type]: The set of unavaialble classes
        """
        return self._unavailable

    @property
    def available_types(self) -> t.Optional[t.List]:
        """Gets the available specialised "types"

        Returns:
            t.Optional[t.List]: The available specialised "types"
        """
        return self._types if hasattr(self, "_types") else None

    def update_registry(self):
        """Updates the factory's registry (available and unavailable production classes)
        """
        self._available = get_concrete_subclasses(self.klass)
        self._unavailable = [k for k in get_subclasses(self.klass) if k not in self._available]

    def match(self, name: str, variant: str = "concrete") -> t.List[t.Dict]:
        """Matches the name to an available production class

        Args:
            name (str): The name to search
            variant (str, optional): The variant (concrete or abstract). Defaults to "concrete".

        Raises:
            TypeError: Wrong type supplied for 'variant'
            ValueError: Wrong value supplied for 'variant'
            AmbiguousMatchError: Ambiguous match among available production classes

        Returns:
            t.List[t.Dict]: A list of dicts with keys: class, name, similarity
        """
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
        """Produces a type that matches the given name

        Args:
            name (str): The name

        Raises:
            NothingMatchedError: Nothing has matched
            AbstractMatchError: The matched type is abstract
            ValueError: Matched with insufficient similarity due to 'name'

        Returns:
            type: The type that matches the name.
        """
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
        """Verify if a type matches the desired type

        Args:
            klass (type): Type to check
        """
        if "_type" in kwargs and hasattr(self, "_type"):
            _type = kwargs.pop("_type")
            _klass_type = getattr(klass, "__type__", None)
            if _klass_type is not _type:
                raise ValueError(
                    f"match '{klass!s}' is of different type ('{_klass_type}') than "
                    f"requested ({_type})"
                )

    def __call__(self, name: str, *args, **kwargs) -> object:
        """Produces an object, main factory method

        Args:
            name (str): The name of the class to create
            *args: Positional arguments passed to the constructor of the class
            **kwargs: Keyword arguments passed to the constructor of the class

        Returns:
            object: The class instance
        """
        klass = self.produce_type(name)
        self.verify_type(klass, **kwargs)
        kwargs.pop("_type", None)
        return klass(*args, **kwargs)

    def make_default(self, *args, **kwargs) -> object:
        """Produces an instance of the default class

        Args:
            *args: Positional arguments passed to the constructor of the default class
            **kwargs: Keyword arguments passed to the constructor of the default class

        Returns:
            object: The default class instance
        """
        klass = self.default
        return klass(*args, **kwargs)
