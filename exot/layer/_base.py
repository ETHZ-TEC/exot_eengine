"""Base class for a layer"""

import abc
import enum
import typing as t

import exot.exceptions
from exot.util.mixins import Configurable, IntermediatesHolder, SubclassTracker


class Layer(
    SubclassTracker, IntermediatesHolder, Configurable, track="layer", metaclass=abc.ABCMeta
):
    """A base class for communication-oriented layers

    Layers that require runtime configuration are considered stateful and must be
    configured before `encode` and `decode` can be called. To indicate whether a
    layer is stateless or stateful, provide the correct boolean value in the property
    `requires_runtime_config`.

    If a layer is stateful and not configured, an exception will be thrown when
    attempting to run the user-facing API `encode`, `decode`. Using raw `_encode` and
    `_decode` might produce undefined behaviour due to the preconditions not being
    satisfied.
    """

    @enum.unique
    class Type(enum.Enum):
        """ Layer types

        The values must be unique, strings, and correspond to keys in the config.
        """

        Line = "lne"
        Source = "src"
        PrePost = "rdp"
        InputOutput = "io"
        Other = "other"

    """
    Validation
    ----------

    To avoid modifying the validation functions below, layers can implement simple
    type checks and value checks using the following properties.

    `_encode_types` and `_decode_types` should produce tuples with only types, such
    that they can be used for checks in `isinstance`.

    `_encode_validators` and `_decode_validators` should produce mappings from types to
    callables. The callables should take a single value and produce a boolean. If
    a type is matched, the callable is called and its output is incorporated into the
    match.
    """

    @property
    def _encode_types(self) -> t.Tuple[type]:
        return tuple()

    @property
    def _decode_types(self) -> t.Tuple[type]:
        return tuple()

    @property
    def _encode_validators(self) -> t.Mapping[type, t.List[t.Callable[[t.Any], bool]]]:
        return dict()

    @property
    def _encode_output_validators(self) -> t.Mapping[type, t.List[t.Callable[[t.Any], bool]]]:
        return self._decode_validators

    @property
    def _decode_validators(self) -> t.Mapping[type, t.List[t.Callable[[t.Any], bool]]]:
        return dict()

    @property
    def _decode_output_validators(self) -> t.Mapping[type, t.List[t.Callable[[t.Any], bool]]]:
        return self._encode_validators

    """
    The validate_{encode,decode}_{input,output} should not return, and only raise
    and exception when an input is invalid (or produce a warning).

    By default, validate_encode_input <==> validate_decode_output, and
    validate_encode_output <==> validate_decode_input, due to the layer-oriented
    architecture. These can be, however, overriden if additional checks are required,
    e.g. dimensions/shape checks.
    """

    def _validate_helper(
        self, value: t.Any, *, encode: bool = True, input: bool = True
    ) -> t.Tuple[bool, t.Dict[type, t.List[bool]]]:
        """Validate types and perform other checks on in/outputs of de/encode"""
        if input:
            _tcheck = self._encode_types if encode else self._decode_types
            _vcheck = self._encode_validators if encode else self._decode_validators
        else:
            _tcheck = self._decode_types if encode else self._encode_types
            _vcheck = (
                self._encode_output_validators if encode else self._decode_output_validators
            )

        if any(not isinstance(_, type) for _ in _vcheck.keys()):
            raise TypeError("invalid keys, validators must have types as keys")

        types_ok: bool = True
        values_ok: t.Dict[type, t.List[bool]] = {k.__name__: [] for k in _vcheck.keys()}

        if _tcheck:
            assert isinstance(_tcheck, tuple) and all(isinstance(_, type) for _ in _tcheck)
            types_ok &= isinstance(value, _tcheck)

        if _vcheck:
            for _type, _validators in _vcheck.items():
                assert isinstance(_type, type), "expects keys to be types"
                for _validator in _validators:
                    assert isinstance(_validator, t.Callable), "expects Callable's"
                    if isinstance(value, _type):
                        values_ok[_type.__name__].append(bool(_validator(value)))

        return types_ok, values_ok

    def _validate_thrower(self, which: str, *args, **kwargs) -> t.NoReturn:
        """Helper to throw errors if type or value checks fail"""
        types_ok, values_ok = self._validate_helper(*args, **kwargs)

        if not types_ok:
            raise exot.exceptions.TypeValidationFailed(
                f"type checks failed for {which} of layer {self.name}",
                type(*args),
                *self._encode_types,
            )

        if not all(all(values_ok[_]) for _ in values_ok):
            raise exot.exceptions.ValueValidationFailed(
                "value checks failed for {} of layer {} (type: checks): {}".format(
                    which, self.name, values_ok
                )
            )

    def validate_encode_input(self, value) -> t.NoReturn:
        self._validate_thrower("encode input", value, encode=True, input=True)

    def validate_encode_output(self, value) -> t.NoReturn:
        self._validate_thrower("encode output", value, encode=True, input=False)

    def validate_decode_input(self, value) -> t.NoReturn:
        self._validate_thrower("decode input", value, encode=False, input=True)

    def validate_decode_output(self, value) -> t.NoReturn:
        self._validate_thrower("decode output", value, encode=False, input=False)

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        """Gets a string representation of the object

        Returns:
            str: The info
        """
        layer_type = self.__type__.name
        configured = "configured" if self.configured else "not configured"
        ereq = "encode requires runtime config" if self.requires_runtime_config[0] else ""
        dreq = "decode requires runtime config" if self.requires_runtime_config[1] else ""

        return (
            f"<{self.name!r} {layer_type} layer at {hex(id(self))}, "
            f"{self.requires_runtime_config}, {configured}>"
        )

    @property
    def requires_runtime_config(self) -> (bool, bool):
        """Does the layer's (encode, decode) require runtime configuration?"""
        return (False, False)

    @abc.abstractmethod
    def _encode(self, encode_input):
        """Abstract method which performs the main encoding operation of the layer

        Due to being wrapped in `encode` the following pre- and post-conditions are
        satisfied:

        - Encoding input is of valid type and/or value
        - Encoding output is of valid type and/or value
        - If runtime configuration is required, the the layer is configured
        """
        pass

    def encode(self, encode_input, skip_checks=False):
        """Encode a stream from a preceding layer"""

        if self.requires_runtime_config[0]:
            if not self.configured:
                raise exot.exceptions.LayerConfigMissing(
                    f"layer {self.__class__.__name__} needs runtime configuration!"
                )

        if not skip_checks:
            self.validate_encode_input(encode_input)
        _ = self._encode(encode_input)
        if not skip_checks:
            self.validate_encode_output(_)
        return _

    @abc.abstractmethod
    def _decode(self, decode_input):
        """Abstract method which performs the main decoding operation of the layer

        Due to being wrapped in `decode` the following pre- and post-conditions are
        satisfied:

        - Decoding input is of valid type and/or value
        - Decoding output is of valid type and/or value
        - If runtime configuration is required, the the layer is configured
        """
        pass

    def decode(self, decode_input, skip_checks=False):
        """Decode a stream from a succeeding layer"""
        if self.requires_runtime_config[1]:
            if not self.configured:
                raise exot.exceptions.LayerConfigMissing(
                    f"layer {self.__class__.__name__} needs runtime configuration!"
                )

        if not skip_checks:
            self.validate_decode_input(decode_input)
        _ = self._decode(decode_input)
        if not skip_checks:
            self.validate_decode_output(_)
        return _
