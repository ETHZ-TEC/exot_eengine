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
"""Classes for generic machine learning data sets, for easier handling of data.
"""

import abc
import enum
import inspect
import math
import os
from pathlib import Path

import numpy as np
import toml

from exot.util.logging import get_root_logger
from exot.util.mixins import Pickleable

__all__ = (
    "DatasetType",
    "GenericDatasetHandler",
    "TrainX",
    "TrainY",
    "TrainSampleLen",
    "TrainWeights",
    "TestX",
    "TestY",
    "TestSampleLen",
    "TestWeights",
    "VerifX",
    "VerifY",
    "VerifSampleLen",
)


@enum.unique
class DatasetType(enum.IntEnum):
    """Stream types"""

    Training = enum.auto()
    Test = enum.auto()
    Verification = enum.auto()


class GenericDataSetHandler(Pickleable, metaclass=abc.ABCMeta):
    @property
    def train_set(self):
        datasets = {}
        for cls in inspect.getmro(type(self)):
            if hasattr(cls, "is_dataset"):
                if cls.is_dataset()[0] == DatasetType.Training:
                    datasets[cls.is_dataset()[1]] = getattr(self, cls.is_dataset()[2])
        return datasets

    @property
    def test_set(self):
        datasets = {}
        for cls in inspect.getmro(type(self)):
            if hasattr(cls, "is_dataset"):
                if cls.is_dataset()[0] == DatasetType.Test:
                    datasets[cls.is_dataset()[1]] = getattr(self, cls.is_dataset()[2])
        return datasets

    @property
    def verification_set(self):
        datasets = {}
        for cls in inspect.getmro(type(self)):
            if hasattr(cls, "is_dataset"):
                if cls.is_dataset()[0] == DatasetType.Verification:
                    datasets[cls.is_dataset()[1]] = getattr(self, cls.is_dataset()[2])
        return datasets

    def remove_dataset(self) -> None:
        """Serialise a Dataset"""
        self.remove_mapping("train_set", path=self.path_dataset, prefix="train_")
        self.remove_mapping("test_set", path=self.path_dataset, prefix="test_")
        self.remove_mapping("verification_set", path=self.path_dataset, prefix="verif_")
        if self.path_dataset.joinpath("_parameters.toml").exists():
            os.remove(self.path_dataset.joinpath("_parameters.toml"))
        get_root_logger().debug(f"serialised dataset from path '{self.path_dataset}' removed")

    def write_dataset(self) -> None:
        """Serialise a Dataset"""
        self.save_mapping("train_set", path=self.path_dataset, prefix="train_")
        self.save_mapping("test_set", path=self.path_dataset, prefix="test_")
        self.save_mapping("verification_set", path=self.path_dataset, prefix="verif_")
        with self.path_dataset.joinpath("_parameters.toml").open("w") as parameterfile:
            toml.dump(self.dataset_parameters_to_dict(), parameterfile)
        get_root_logger().debug(f"serialised dataset to '{self.path_dataset}'")

    def read_dataset(self) -> None:
        """Deserialise a Dataset"""
        train_set = self.load_mapping(path=self.path_dataset, prefix="train_")
        for key in train_set:
            setattr(self, "train_" + key, train_set[key])
        test_set = self.load_mapping(path=self.path_dataset, prefix="test_")
        for key in test_set:
            setattr(self, "test_" + key, test_set[key])
        verif_set = self.load_mapping(path=self.path_dataset, prefix="verif_")
        for key in verif_set:
            setattr(self, "verif_" + key, verif_set[key])
        with self.path_dataset.joinpath("_parameters.toml").open("r") as parameterfile:
            param_dict = toml.load(parameterfile)
        self.dataset_parameters_from_dict(param_dict)
        get_root_logger().debug(f"serialised dataset to '{self.path_dataset}'")

    @property
    @abc.abstractmethod
    def path_dataset(self) -> Path:
        pass

    @property
    @abc.abstractmethod
    def batch_size(self):
        pass

    @batch_size.setter
    @abc.abstractmethod
    def batch_size(self, value):
        pass

    @property
    def num_feature_dims(self):
        return self.train_X_shape[-1]

    @num_feature_dims.setter
    @abc.abstractmethod
    def num_feature_dims(self, value):
        pass

    @property
    def num_train_batches(self):
        if not hasattr(self, "_num_train_batches"):
            self._num_train_batches = None
        return self._num_train_batches

    @num_train_batches.setter
    def num_train_batches(self, value: int):
        self._num_train_batches = value

    @property
    def num_test_batches(self):
        if not hasattr(self, "_num_test_batches"):
            self._num_test_batches = None
        return self._num_test_batches

    @num_test_batches.setter
    def num_test_batches(self, value: int):
        self._num_test_batches = value

    @property
    def num_verif_batches(self):
        if not hasattr(self, "_num_verif_batches"):
            self._num_verif_batches = None
        return self._num_verif_batches

    @num_verif_batches.setter
    def num_verif_batches(self, value: int):
        self._num_verif_batches = value

    def dataset_parameters_to_dict(self) -> dict:
        return dict(
            batch_size=self.batch_size,
            num_feature_dims=self.num_feature_dims,
            num_train_batches=self.num_train_batches,
            num_test_batches=self.num_test_batches,
            num_verif_batches=self.num_verif_batches,
        )

    def dataset_parameters_from_dict(self, param_dict: dict) -> None:
        not_set = list()
        for key in param_dict:
            try:
                setattr(self, key, param_dict[key])
            except:
                # This might happen as some attributes don't have a setter. No big deal, but the developer should be informed.
                not_set.append(key)
        if len(not_set) > 0:
            get_root_logger().warning(f"Following attributes could not be set: {not_set}")


class TrainX(metaclass=abc.ABCMeta):
    @staticmethod
    def is_dataset():
        return (DatasetType.Training, "X", "train_X")

    @property
    def train_X_shape(self):
        if hasattr(self, "_train_X"):
            return self._train_X.shape
        else:
            return None

    @property
    def train_X(self):
        return getattr(self, "_train_X", None)

    @train_X.setter
    def train_X(self, value):
        if isinstance(value, np.ndarray):
            self._train_X = value.astype("float32")
        else:
            raise ValueError(f"Wrong type for X, is {type(value)} but should be numpy.ndarray.")


class TrainY(metaclass=abc.ABCMeta):
    @staticmethod
    def is_dataset():
        return (DatasetType.Training, "Y", "train_Y")

    @property
    def train_Y_shape(self):
        if hasattr(self, "_train_Y"):
            return self._train_Y.shape
        else:
            return None

    @property
    def train_Y(self):
        return getattr(self, "_train_Y", None)

    @train_Y.setter
    def train_Y(self, value):
        if isinstance(value, np.ndarray):
            self._train_Y = value.astype("int32")
        else:
            raise ValueError(f"Wrong type for Y, is {type(value)} but should be numpy.ndarray.")


class TrainSampleLen(metaclass=abc.ABCMeta):
    @staticmethod
    def is_dataset():
        return (DatasetType.Training, "actual_batch_len", "train_actual_batch_len")

    @property
    def train_actual_batch_len_shape(self):
        if hasattr(self, "_train_actual_batch_len"):
            return self._train_actual_batch_len.shape
        else:
            return None
        pass

    @property
    def train_actual_batch_len(self):
        return getattr(self, "_train_actual_batch_len", None)

    @train_actual_batch_len.setter
    def train_actual_batch_len(self, value):
        if isinstance(value, np.ndarray):
            self._train_actual_batch_len = value.astype("int32")
        else:
            raise ValueError(
                f"Wrong type for train_actual_batch_len, is {type(value)} but should be numpy.ndarray."
            )


class TrainWeights(metaclass=abc.ABCMeta):
    @staticmethod
    def is_dataset():
        return (DatasetType.Training, "weights", "train_weights")

    @property
    def train_weights_shape(self):
        if hasattr(self, "_train_weights"):
            return self._train_weights.shape
        else:
            return None
        pass

    @property
    def train_weights(self):
        return getattr(self, "_train_weights", None)

    @train_weights.setter
    def train_weights(self, value):
        if isinstance(value, np.ndarray):
            self._train_weights = value.astype("float32")
        else:
            raise ValueError(
                f"Wrong type for weights, is {type(value)} but should be numpy.ndarray."
            )


class TestX(metaclass=abc.ABCMeta):
    @staticmethod
    def is_dataset():
        return (DatasetType.Test, "X", "test_X")

    @property
    def test_X_shape(self):
        if hasattr(self, "_test_X"):
            return self._test_X.shape
        else:
            return None
        pass

    @property
    def test_X(self):
        return getattr(self, "_test_X", None)

    @test_X.setter
    def test_X(self, value):
        if isinstance(value, np.ndarray):
            self._test_X = value.astype("float32")
        else:
            raise ValueError(f"Wrong type for X, is {type(value)} but should be numpy.ndarray.")


class TestY(metaclass=abc.ABCMeta):
    @staticmethod
    def is_dataset():
        return (DatasetType.Test, "Y", "test_Y")

    @property
    def test_Y_shape(self):
        if hasattr(self, "_test_Y"):
            return self._test_Y.shape
        else:
            return None
        pass

    @property
    def test_Y(self):
        return getattr(self, "_test_Y", None)

    @test_Y.setter
    def test_Y(self, value):
        if isinstance(value, np.ndarray):
            self._test_Y = value.astype("int32")
        else:
            raise ValueError(f"Wrong type for Y, is {type(value)} but should be numpy.ndarray.")


class TestSampleLen(metaclass=abc.ABCMeta):
    @staticmethod
    def is_dataset():
        return (DatasetType.Test, "actual_batch_len", "test_actual_batch_len")

    @property
    def test_actual_batch_len_shape(self):
        if hasattr(self, "_test_actual_batch_len"):
            return self._test_actual_batch_len.shape
        else:
            return None
        pass

    @property
    def test_actual_batch_len(self):
        return getattr(self, "_test_actual_batch_len", None)

    @test_actual_batch_len.setter
    def test_actual_batch_len(self, value):
        if isinstance(value, np.ndarray):
            self._test_actual_batch_len = value.astype("int32")
        else:
            raise ValueError(
                f"Wrong type for actual_batch_len, is {type(value)} but should be numpy.ndarray."
            )


class TestWeights(metaclass=abc.ABCMeta):
    @staticmethod
    def is_dataset():
        return (DatasetType.Test, "weights", "test_weights")

    @property
    def test_weights_shape(self):
        if hasattr(self, "_test_weights"):
            return self._test_weights.shape
        else:
            return None
        pass

    @property
    def test_weights(self):
        return getattr(self, "_test_weights", None)

    @test_weights.setter
    def test_weights(self, value):
        if isinstance(value, np.ndarray):
            self._test_weights = value.astype("float32")
        else:
            raise ValueError(
                f"Wrong type for weights, is {type(value)} but should be numpy.ndarray."
            )


class VerifX(metaclass=abc.ABCMeta):
    @staticmethod
    def is_dataset():
        return (DatasetType.Verification, "X", "verif_X")

    @property
    def verif_X_shape(self):
        if hasattr(self, "_verif_X"):
            return self._verif_X.shape
        else:
            return None
        pass

    @property
    def verif_X(self):
        return getattr(self, "_verif_X", None)

    @verif_X.setter
    def verif_X(self, value):
        if isinstance(value, np.ndarray):
            self._verif_X = value.astype("float32")
        else:
            raise ValueError(
                f"Wrong type for verif_X, is {type(value)} but should be numpy.ndarray."
            )


class VerifY(metaclass=abc.ABCMeta):
    @staticmethod
    def is_dataset():
        return (DatasetType.Verification, "Y", "verif_Y")

    @property
    def verif_Y_shape(self):
        if hasattr(self, "_verif_Y"):
            return self._verif_Y.shape
        else:
            return None
        pass

    @property
    def verif_Y(self):
        return getattr(self, "_verif_Y", None)

    @verif_Y.setter
    def verif_Y(self, value):
        if isinstance(value, np.ndarray):
            self._verif_Y = value.astype("int32")
        else:
            raise ValueError(
                f"Wrong type for verif_Y, is {type(value)} but should be numpy.ndarray."
            )


class VerifSampleLen(metaclass=abc.ABCMeta):
    @staticmethod
    def is_dataset():
        return (DatasetType.Verification, "actual_batch_len", "verif_actual_batch_len")

    @property
    def verif_actual_batch_len_shape(self):
        if hasattr(self, "_verif_actual_batch_len"):
            return self._verif_actual_batch_len.shape
        else:
            return None
        pass

    @property
    def verif_actual_batch_len(self):
        return getattr(self, "_verif_actual_batch_len", None)

    @verif_actual_batch_len.setter
    def verif_actual_batch_len(self, value):
        if isinstance(value, np.ndarray):
            self._verif_actual_batch_len = value.astype("int32")
        else:
            raise ValueError(
                f"Wrong type for verif_actual_batch_len, is {type(value)} but should be numpy.ndarray."
            )
