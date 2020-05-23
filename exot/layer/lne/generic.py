"""Generic line-coding layer"""

import copy as cp
import typing as t

import numpy as np
import pandas as pd
import scipy.interpolate
import sklearn.base
import sklearn.naive_bayes

from exot.exceptions import *
from exot.layer.src.huffman import Huffman
from exot.util.misc import dict_depth
from exot.util.scinum import is_fitted

from .._base import Layer


"""
GenericLineCoding
-----------

The GenericLineCoding layer performs the following:

-   [Encoding] With a signal mapping provided in the constructor, applies the mapping to
    an input symbol stream using keys from the mapping, and produces either:

    -   a flattened 1-d stream of subsymbols,
    -   a 2-d array with shape: (symbol stream length, subsymbols * samples/subsymbol)

    The type of output is controlled with a 'flattened' parameter. An unflattened
    version can always be produced with the `_encode_as_subsymbols` method. Since
    it's more meaningful for this layer to return 1-d streams, the condition that
    decode(encode(x)) == x is not valid by default. The `_encode_as_subsymbols` can
    be used for dummy input or testing.

-   [Decoding] Takes a 2-d array of samples, where each row should contain a number of
    samples meant to represent a single symbol, i.e. should contain all subsymbols. A
    mask is used with phase-offseted 'carrier' symbols.

    The symbol space contains a number of phases equal to: 2 * ("subsymbol count" - 1).
    For simple 2-valued Manchester coding, these would form 2 phases, with a 90° shift:
    0: [1, 1, 0, 0], 90: [0, 1, 1, 0]. The NumPy `rotate` function is used to produce
    the permutations of the carrier for the mask. Masks are interpolated to get the
    dimensions to match the input. The lower limit is that the number of samples should
    be at least twice the number of subsamples.

Example mappings and masks:
>>> signal = {0: [1, 0], 1: [0, 1], "carrier": [1, 0]}
>>> coder = GenericLineCoding(signal=signal)
>>> coder.create_mask()
array([[1., 1., 0., 0.],
       [0., 1., 1., 0.]])
>>> coder.signal = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], "carrier": [1, 0, 0]}
>>> coder.create_mask()
array([[1., 1., 0., 0., 0., 0.],
       [0., 1., 1., 0., 0., 0.],
       [0., 0., 1., 1., 0., 0.],
       [0., 0., 0., 1., 1., 0.]])
>>> coder.signal = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], "carrier": [0.5, 0.2, 0.5]}
array([[0.5, 0.5, 0.2, 0.2, 0.5, 0.5],
       [0.5, 0.5, 0.5, 0.2, 0.2, 0.5],
       [0.5, 0.5, 0.5, 0.5, 0.2, 0.2],
       [0.2, 0.5, 0.5, 0.5, 0.5, 0.2]])
>>> coder.signal = {0: [1, 0], 1: [0, 1], "carrier": [0.7, 1, 1, 0.7, 0, 0, 0, 0]}
>>> from numpy import set_printoptions; set_printoptions(precision=2)
>>> coder.create_mask(samples=5, kind="linear")
array([[0.7 , 0.93, 1.  , 0.9 , 0.62, 0.08, 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.7 , 0.93, 1.  , 0.9 , 0.62, 0.08, 0.  , 0.  ]])

Constructor arguments:
- "signal" (required) the signal definition, with integer symbols and a 'carrier' key,
- "samples_per_subsymbol" (optional, default: 2),
- "flattened" (optional, default: True).

Preconditions:
- Encode input is a NumPy array,
- Encode input is a 1-d stream,
- Encode input has only values defined in the signal mapping (signal sans carrier),
- Decode input is a 2-d array of valid width.

Postconditions:
- Encode output is a 1-d stream or a 2-d array of valid width,
- Decode output is a 1-d stream,
- Decode output has only values defined in the signal mapping.
"""


class GenericLineCoding(Layer, layer=Layer.Type.Line):
    @property
    def _encode_types(self) -> t.Tuple[type]:
        return (np.ndarray,)

    @property
    def _decode_types(self) -> t.Tuple[type]:
        return (np.ndarray,)

    @property
    def _encode_validators(self) -> t.Mapping[type, t.Callable[[object], bool]]:
        return {
            np.ndarray: [lambda v: v.ndim == 1, lambda v: np.isin(v, list(self.mapping)).all()]
        }

    @property
    def _decode_validators(self) -> t.Mapping[type, t.Callable[[object], bool]]:
        return {
            np.ndarray: [
                lambda v: (v.ndim == 1)
                or (v.ndim == 2 and v.shape[1] >= self.subsymbol_count * 2)
                or (v.ndim == 3)
            ]
        }

    @property
    def required_config_keys(self) -> t.List[str]:
        return ["symstream"]

    def validate(self) -> t.NoReturn:
        """Implements method `validate` from Configurable

        Raises:
            LayerMisconfigured: If wrong symstream was provided
        """
        _ = self.config.symstream

        if not isinstance(_, np.ndarray):
            raise LayerMisconfigured("symstream not an array", type(_), np.ndarray)

        if not (_.ndim == 1) or (_.ndim == 2 and _.shape[1] == 1):
            raise LayerMisconfigured(f"symstream not trivial to flatten: ({_.shape})")

    @property
    def requires_runtime_config(self) -> bool:
        """Decoding needs access to 'ideal_symstream'"""
        return (False, True)

    def __init__(
        self,
        *args,
        signal: t.Mapping,
        samples_per_subsymbol: int = 2,
        demean: bool = True,
        flattened: bool = True,
        saturated: bool = False,
        decode_flattened: bool = False,
        roll_divisions: int = 2,
        min_samples: int = 2,
        symbol_space_op: t.Callable = np.sum,
        saturate_mapping: t.Mapping = {1: -1},
        **kwargs,
    ):
        """Initialise the GenericLineCoding object

        Args:
            *args: passthrough...
            signal (t.Mapping): a signal mapping
            samples_per_subsymbol (int, optional): default number of samples/subsymbol
            flattened (bool, optional): operates in flattened mode?
            saturated (bool, optional): saturate the output?
            symbol_to_saturate (int, optional): symbol to saturate, e.g. 1
            saturate_replacement_symbol (int, optional): replacement symbol, e.g. -1
            **kwargs: passthrough...
        """
        self.signal = signal
        self.samples_per_subsymbol = samples_per_subsymbol
        self.demean = demean
        self.flattened = flattened
        self.saturated = saturated
        self.saturate_mapping = saturate_mapping
        self.decode_flattened = decode_flattened
        self.roll_divisions = roll_divisions
        self.min_samples = min_samples
        self.symbol_space_op = symbol_space_op
        super().__init__(*args, **kwargs)

    @property
    def decode_flattened(self):
        return getattr(self, "_decode_flattened", False)

    @property
    def roll_divisions(self):
        return getattr(self, "_roll_divisions", 2)

    @property
    def min_samples(self):
        return getattr(self, "_min_samples", 2)

    @property
    def symbol_space_op(self):
        return getattr(self, "_symbol_space_op", np.sum)

    @decode_flattened.setter
    def decode_flattened(self, value):
        if not isinstance(value, bool):
            raise LayerTypeError("Wrong type supplied to 'decode_flattened'", type(value), bool)

        self._decode_flattened = value

        if value:
            raise DeprecationWarning(
                "decode_flattened breaks the layered structure and will be removed"
            )

    @roll_divisions.setter
    def roll_divisions(self, value):
        if not isinstance(value, (int, np.integer)):
            raise LayerTypeError("Wrong type supplied to 'roll_divisions'", type(value), int)
        if not value >= 1:
            raise LayerValueError("'roll_divisions' must be equal or greater than 1")

        self._roll_divisions = value

    @min_samples.setter
    def min_samples(self, value):
        if not isinstance(value, (int, np.integer)):
            raise LayerTypeError("Wrong type supplied to 'min_samples'", type(value), int)
        if not value > 0:
            raise LayerValueError("'min_samples' must be greater than 0")

        self._min_samples = value

    @symbol_space_op.setter
    def symbol_space_op(self, value):
        if not isinstance(value, t.Callable):
            raise LayerTypeError(
                "Wrong type supplied to 'symbol_space_op'", type(value), t.Callable
            )

        try:
            dummy = np.eye(3)
            _ = value(dummy, 1)
        except (TypeError, AttributeError) as e:
            raise LayerValueError(
                "provided 'symbol_space_op' does not conform to a np.mean/np.sum-like interface"
            )

        self._symbol_space_op = value

    @property
    def saturate_mapping(self) -> t.Mapping[int, int]:
        return getattr(self, "_saturate_mapping", None)

    @saturate_mapping.setter
    def saturate_mapping(self, value: t.Mapping) -> None:
        try:
            assert isinstance(value, t.Mapping), "must be a mapping"
            assert value, "must not be empty"
            assert dict_depth(value) == 1, "must be a mapping of depth 1"
            assert all(isinstance(_, int) for _ in value.values()), "must have int values"
        except AssertionError as e:
            raise LayerMisconfigured("saturate_mapping {}, got {}".format(*e.args, value))

        if any(isinstance(_, str) for _ in value.keys()):
            _ = {}
            for k, v in value.items():
                _[int(k)] = v
            value = _

        self._saturate_mapping = dict(sorted(value.items(), key=lambda i: i[0]))

    @property
    def signal(self) -> t.Mapping:
        """Get the signal mapping

        Returns:
            t.Mapping: a mapping, same as Channel's signal
        """
        return self._signal

    @signal.setter
    def signal(self, value: t.Mapping) -> None:
        """Set the signal mapping

        Args:
            value (t.Mapping): a mapping with symbols and a 'carrier'

        Raises:
            LayerMisconfigured: If incompatible value is provided
        """
        try:
            assert isinstance(value, t.Mapping), "must be a mapping"
            assert "carrier" in value, "must have a 'carrier' key"
            mapping = dict(filter(lambda x: isinstance(x[0], int), value.items()))
            _0, *_1 = mapping.values()
            assert all(len(_) == len(_0) for _ in _1), "must have same-length mappings"

            self._mapping = mapping
            self._carrier = value["carrier"]
            self._signal = value
            self._subsymbol_count = len(_0)

            if self._subsymbol_count < 2:
                raise LayerMisconfigured(
                    "generic line mapping works only on signals with at " "least 2 subsymbols"
                )

        except AssertionError as e:
            raise LayerMisconfigured("signal {}, got {}".format(*e.args, value))

    @property
    def mapping(self) -> t.Mapping:
        """Gets the symbol-only mapping (signal sans 'carrier')"""
        return self._mapping

    @property
    def demean(self):
        return self._demean

    @demean.setter
    def demean(self, value):
        if not isinstance(value, bool):
            raise LayerMisconfigured("Demean must be a boolean value")

        self._demean = value

    @property
    def carrier(self) -> t.List:
        """Gets the carrier"""
        return self._carrier

    @property
    def flattened(self) -> bool:
        """Should the output be flattened?"""
        return self._flattened

    @flattened.setter
    def flattened(self, value) -> None:
        """Sets the 'flattened' property"""
        if not isinstance(value, bool):
            raise LayerMisconfigured(f"'flattened' must be a bool, got {type(value)}")
        self._flattened = value

    @property
    def saturated(self) -> bool:
        """Should the output be saturated?"""
        return self._saturated

    @saturated.setter
    def saturated(self, value) -> None:
        """Sets the 'saturated' property"""
        if not isinstance(value, bool):
            raise LayerMisconfigured(f"'saturated' must be a bool, got {type(value)}")
        self._saturated = value

    @property
    def max_num_symbols(self) -> int:
        return self._subsymbol_count

    def symrate_to_subsymrate(self, symrate: t.T) -> t.T:
        """Convert a symbol rate to a subsymbol rate"""
        return symrate * self.subsymbol_count

    @property
    def subsymbol_count(self) -> int:
        """Get the number of subsymbols"""
        return self._subsymbol_count

    @property
    def phase_count(self) -> int:
        return 2 * (self.subsymbol_count - 1)

    @property
    def samples_per_subsymbol(self) -> int:
        return self._samples_per_subsymbol

    @property
    def decision_device(self) -> t.Optional[object]:
        return getattr(self, "_decision_device", None)

    @decision_device.setter
    def decision_device(self, value) -> None:
        if isinstance(value, type):
            raise LayerMisconfigured("'decision_device' must be an object, not a type")
        if not sklearn.base.is_classifier(value):
            raise LayerMisconfigured("'decision_device' can only be a classifier")

        self._decision_device = value

    @samples_per_subsymbol.setter
    def samples_per_subsymbol(self, value: int) -> None:
        try:
            assert isinstance(value, int), ("must be an int", type(value))
            assert value != 0, ("must be non-zero", value)
            self._samples_per_subsymbol = value
        except AssertionError as e:
            raise LayerMisconfigured("samples_per_subsymbol {}, got {}".format(*e.args))

    def is_compatible(self, num_symbols: int) -> bool:
        return self.max_num_symbols >= num_symbols

    def _encode_as_subsymbols(self, symstream: np.ndarray, *, samples: int = 1) -> np.ndarray:
        """Encode a symbol stream as an array of subsymbols

        Args:
            symstream (np.ndarray): a symbol stream
            samples (int, optional): number of samples per symbol

        Returns:
            np.ndarray: a 2-d array of shape: len(symstream), subsymbol_count * samples
        """
        shape = (len(symstream), self.subsymbol_count)
        lnestream = np.array([self.signal[_] for _ in symstream], dtype=np.dtype("int"))
        assert lnestream.shape == shape, "generated stream must have the desired shape"
        if samples > 1:
            return np.repeat(lnestream, samples, axis=1)
        else:
            return lnestream

    def _apply_saturate_mapping(self, value: np.ndarray) -> np.ndarray:
        return np.vectorize(
            lambda v: self.saturate_mapping[v] if v in self.saturate_mapping else v
        )(value)

    def _encode(self, symstream: np.ndarray) -> np.ndarray:
        """Encode a symbol stream

        Args:
            symstream (np.ndarray): a symbol stream

        Returns:
            np.ndarray: the encoded (and/or flattened, and/or saturated) stream
        """
        if self.flattened:
            _ = self._encode_as_subsymbols(symstream, samples=1)
        else:
            _ = self._encode_as_subsymbols(symstream, samples=self.samples_per_subsymbol)

        if self.saturated:
            _ = self._apply_saturate_mapping(_)

        return _.flatten() if self.flattened else _

    def create_mask(self, *, samples: int = -1, kind: str = "nearest") -> np.ndarray:
        """Create a mask for symbol space computation

        Args:
            samples (int, optional): number of samples per subsymbol
            kind (str, optional): interpolation method

        Returns:
            np.ndarray: a mask of shape: subsymbol_count * samples, phase_count
        """
        samples = self.samples_per_subsymbol if samples == -1 else samples
        samples = self.min_samples if samples < self.min_samples else samples

        _target_length = self.subsymbol_count * samples
        _interpolator = scipy.interpolate.interp1d(
            x=np.linspace(0, 1, len(self.carrier)), y=self.carrier, kind=kind
        )
        _carrier = _interpolator(np.linspace(0, 1, _target_length))
        _roll = samples // self.roll_divisions

        return np.array([np.roll(_carrier, _roll * idx) for idx in range(self.phase_count)])

    def create_symbol_space(self, data: np.ndarray) -> np.ndarray:
        """Create a symbol space representation of input data

        Args:
            data (np.ndarray): a 2-d array

        Returns:
            np.ndarray: a symbol space or shape: data.shape[0], phase_count
        """
        _samples_per_subsymbol = data.shape[1] // self.subsymbol_count
        _symbol_space = np.zeros((data.shape[0], self.phase_count))
        if self.demean:
            data = data - np.mean(data, axis=1, keepdims=True)
        _mask = self.create_mask(samples=_samples_per_subsymbol)

        for phase in range(self.phase_count):
            _symbol_space[:, phase] = self.symbol_space_op(
                np.multiply(data, _mask[phase, :]), axis=1
            )

        return _symbol_space

    @property
    def symbol_space(self) -> t.Optional[np.ndarray]:
        return getattr(self, "_symbol_space", None)

    def _decode(self, lnestream: np.ndarray) -> np.ndarray:
        """Decode an input line-encoded stream

        Args:
            lnestream (np.ndarray): a 2-d array of width at least 2 * subsymbol_count

        Returns:
            np.ndarray: a predicted symstream
        """
        if lnestream.ndim == 3:
            lnestream = np.vstack(np.flip(lnestream, axis=1))

        if lnestream.ndim != 2:
            raise LayerValueError("Wrong number of dimensions in the supplied lnestream")

        ideal: np.ndarray

        if self.decode_flattened:
            raise DeprecationWarning(
                "decode_flattened breaks the layered structure and will be removed"
            )

            self._symbol_space = self.create_symbol_space(lnestream).flatten()
            ideal = self.config.bitstream

            self._symbol_space = self._symbol_space[slice(None, len(ideal))]
            ideal = ideal[slice(None, len(self._symbol_space))]
            self._symbol_space = self._symbol_space.reshape(-1, 1)
        else:
            self._symbol_space = self.create_symbol_space(lnestream)
            ideal = self.config.symstream

            self._symbol_space = self._symbol_space[slice(None, len(ideal)), :]
            ideal = ideal[slice(None, len(self._symbol_space))]

        self.add_intermediate("symbol_space", self._symbol_space)

        if "decision_device" in self.config:
            self.decision_device = self.config.decision_device
        else:
            self.decision_device = sklearn.naive_bayes.GaussianNB()

        if not is_fitted(self.decision_device):
            self.decision_device.fit(self.symbol_space, ideal)

        predictions = self.decision_device.predict(self.symbol_space)
        self.add_intermediate(
            "decision_device",
            pd.DataFrame({"decision_device": [cp.copy(self.decision_device)]}),
        )

        return Huffman(length=4).encode(predictions) if self.decode_flattened else predictions
