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
"""Apply thermal data augmentation algorithm to generate a big training and test dataset from a smaller set of measurement traces."""

import abc
import enum
import math
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import toml
from sklearn.utils import shuffle

from exot.channel._base import Analysis
from exot.channel.mixins.mldataset import (
    TestSampleLen,
    TestWeights,
    TestX,
    TestY,
    TrainSampleLen,
    TrainWeights,
    TrainX,
    TrainY,
    VerifSampleLen,
    VerifX,
    VerifY,
)
from exot.exceptions import *
from exot.util.attributedict import AttributeDict, LabelMapping
from exot.util.logging import get_root_logger
from exot.util.misc import validate_helper
from exot.util.wrangle import Matcher

from ._mixins import ThermalAppDecectionDataSetHandler

__all__ = "DataAugmentation"


@enum.unique
class SampleSelectionMode(enum.Enum):
    """ Layer types

    The values must be unique, strings, and correspond to keys in the config.
    """

    Random = "random"
    Uniform = "uniform"
    Alternating = "alternating"


@enum.unique
class WeightCalculationMode(enum.Enum):
    """ Layer types

    The values must be unique, strings, and correspond to keys in the config.
    """

    Equal = "equal"
    Proportional = "proportional"
    Binary = "binary"


class DataAugmentation(
    Analysis,
    ThermalAppDecectionDataSetHandler,
    TrainX,
    TrainY,
    TrainSampleLen,
    TrainWeights,
    TestX,
    TestY,
    TestSampleLen,
    TestWeights,
    VerifX,
    VerifY,
    VerifSampleLen,
):
    def __init__(self, *args, **kwargs):
        Analysis.__init__(self, *args, **kwargs)

        validate = partial(validate_helper, self.config, msg="Analysis")

        # top-level configuration
        validate("name", str)
        validate("type", str)

        validate(("DATASET"), AttributeDict)

        validate(("DATASET", "env"), (str))

        for matcher_name in ["matcher_data", "matcher_labels"]:
            validate(("DATASET", matcher_name), (dict, AttributeDict))
            assert len(getattr(self.config.DATASET, matcher_name)) == 4
            assert isinstance(getattr(self.config.DATASET, matcher_name)["quantity"], str)
            assert isinstance(getattr(self.config.DATASET, matcher_name)["method"], str)
            assert isinstance(getattr(self.config.DATASET, matcher_name)["values"], list)
            assert all(
                isinstance(n, str) for n in getattr(self.config.DATASET, matcher_name)["values"]
            )
            assert isinstance(getattr(self.config.DATASET, matcher_name)["dimensions"], list)
            assert all(
                isinstance(n, int)
                for n in getattr(self.config.DATASET, matcher_name)["dimensions"]
            )

        validate(("DATASET", "ambient"), (list))
        assert len(self.config.DATASET.ambient) == 2
        assert all(isinstance(n, int) for n in self.config.DATASET.ambient)

        validate(("DATASET", "num_test_batches"), (int))
        validate(("DATASET", "num_train_batches"), (int))
        validate(("DATASET", "batch_size_s"), (int))
        validate(("DATASET", "min_len_profile_s"), (int, float))
        validate(("DATASET", "train_phases"), (list))
        assert all(isinstance(n, str) for n in self.config.DATASET.train_phases)
        validate(("DATASET", "test_phases"), (list))
        assert all(isinstance(n, str) for n in self.config.DATASET.test_phases)
        validate(("DATASET", "verif_phases"), (list))
        assert all(isinstance(n, str) for n in self.config.DATASET.verif_phases)
        validate(("DATASET", "resampling_period_s"), (int, float))
        validate(("DATASET", "augmentation_train"), (bool))
        validate(("DATASET", "augmentation_test"), (bool))

        validate(("DATASET", "label_mapping"), AttributeDict)

    def _get_runs(self, analysis_phase: str):
        if not hasattr(self, "_" + analysis_phase + "_runs"):
            setattr(self, "_" + analysis_phase + "_runs", [])
            for meas_phase in getattr(self, analysis_phase + "_phases"):
                for cur_run in self.experiment.phases[meas_phase].values():
                    getattr(self, "_" + analysis_phase + "_runs").append(
                        (
                            cur_run,
                            self.experiment.config.EXPERIMENT.PHASES[meas_phase].repetitions,
                        )
                    )
        return getattr(self, "_" + analysis_phase + "_runs")

    @property
    def train_runs(self):
        return self._get_runs("train")

    @property
    def test_runs(self):
        return self._get_runs("test")

    @property
    def verif_runs(self):
        return self._get_runs("verif")

    @property
    def train_phases(self):
        return self.config.DATASET.train_phases

    @property
    def test_phases(self):
        return self.config.DATASET.test_phases

    @property
    def verif_phases(self):
        return self.config.DATASET.verif_phases

    @property
    def ambient(self):
        return self.config.DATASET.ambient

    @property
    def dynamic_start(self):
        return getattr(self.config.DATASET, "dynamic_start", (0, 0))

    @property
    def thermal_throttling(self):
        return getattr(self.config.DATASET, "thermal_throttling", np.inf)

    @property
    def min_len_profile(self):
        return math.ceil(self.config.DATASET.min_len_profile_s / self.resampling_period_s)

    @property
    def duration_context_change(self):
        return math.ceil(
            self.config.DATASET.duration_context_change_s / self.resampling_period_s
        )

    def _populate_columns_idxes(self):
        cur_run = random.choice(self.train_runs)[0]
        dummy_data = pd.read_csv(cur_run.rep_path(self.env, 0).joinpath("snk.log.csv"))
        self._data_columns_idxes = list(range(1, dummy_data[self.matcher_data].shape[1]))
        self._label_columns_idxes = list(
            range(
                dummy_data[self.matcher_data].shape[1],
                dummy_data[self.matcher_data].shape[1]
                + dummy_data[self.matcher_labels].shape[1]
                - 1,
            )
        )
        self._beta_z_names = list(dummy_data[self.matcher_data].columns[1:])

    @property
    def beta_z_names(self):
        if not hasattr(self, "_beta_z_names"):
            self._populate_columns_idxes()
        return self._beta_z_names

    @property
    def data_columns_idxes(self):
        if not hasattr(self, "_data_columns_idxes"):
            self._populate_columns_idxes()
        return self._data_columns_idxes

    @property
    def label_columns_idxes(self):
        if not hasattr(self, "_label_columns_idxes"):
            self._populate_columns_idxes()
        return self._label_columns_idxes

    @property
    def resampling_period_s(self):
        return self.config.DATASET.resampling_period_s

    @property
    def env(self):
        return self.config.DATASET.env

    @property
    def matcher_data(self):
        if not hasattr(self, "_matcher_data"):
            self._matcher_data = Matcher(
                self.config.DATASET.matcher_data["quantity"],
                self.config.DATASET.matcher_data["method"],
                self.config.DATASET.matcher_data["values"],
                self.config.DATASET.matcher_data["dimensions"],
            )
        return self._matcher_data

    @property
    def matcher_labels(self):
        if not hasattr(self, "_matcher_labels"):
            self._matcher_labels = Matcher(
                self.config.DATASET.matcher_labels["quantity"],
                self.config.DATASET.matcher_labels["method"],
                self.config.DATASET.matcher_labels["values"],
                self.config.DATASET.matcher_labels["dimensions"],
            )
        return self._matcher_labels

    @property
    def num_train_batches(self):
        return self.config.DATASET.num_train_batches

    @num_train_batches.setter
    def num_train_batches(self, value: int) -> None:
        self.config.DATASET.num_train_batches = value

    @property
    def num_test_batches(self):
        return self.config.DATASET.num_test_batches

    @num_test_batches.setter
    def num_test_batches(self, value: int) -> None:
        self.config.DATASET.num_test_batches = value

    @property
    def num_verif_batches(self):
        if not hasattr(self.config.DATASET, "num_verif_batches"):
            self.config.DATASET.num_verif_batches = len(self.verif_runs)
        return self.config.DATASET.num_verif_batches

    @num_verif_batches.setter
    def num_verif_batches(self, value: int) -> None:
        self.config.DATASET.num_verif_batches = value

    @property
    def augmentation_verif(self):
        return getattr(self.config.DATASET, "augmentation_verif", False)

    @property
    def augmentation_test(self):
        return getattr(self.config.DATASET, "augmentation_test", False)

    @property
    def augmentation_train(self):
        return getattr(self.config.DATASET, "augmentation_train", False)

    @property
    def label_mapping(self):
        if not hasattr(self, "_label_mapping"):
            self._label_mapping = LabelMapping(self.config.DATASET.label_mapping)
        return self._label_mapping

    @property
    def num_labels(self):
        if self.unknown_label:
            return self.label_mapping["UNKNOWN"]["int"] + 1
        else:
            return self.label_mapping["UNKNOWN"]["int"]

    def label_refactoring_rule(self, analysis_phase: str = ""):
        rule_dict = getattr(self.config.DATASET, "label_refactoring_rule", {})
        if hasattr(self.config.DATASET.label_refactoring_rule, "remove_unknown"):
            if getattr(
                self.config.DATASET.label_refactoring_rule.remove_unknown, analysis_phase, False
            ):
                rule_dict["remove_labels"] = [self.label_mapping["UNKNOWN"]]
        return rule_dict

    def ingest_args(self, rep: int, analysis_phase: str) -> dict:
        return dict(
            env=self.env,
            io={
                "rep": rep,
                "matcher": [(self.matcher_data, None), (self.matcher_labels, None)],
                "synchronise": False,
            },
            rdp={"mapping": (self.matcher_labels, self.label_mapping)},
            lne={
                "beta_z_names": self.beta_z_names,
                "label_columns": self.label_columns_idxes,
                "options": self.label_refactoring_rule(analysis_phase),
            },
        )

    @property
    def unknown_label(self) -> bool:
        remove_unknown_label = False
        if hasattr(self.config.DATASET.label_refactoring_rule, "remove_unknown"):
            for _, flag in self.config.DATASET.label_refactoring_rule.remove_unknown.items():
                remove_unknown_label = remove_unknown_label or flag
        return not remove_unknown_label

    @unknown_label.setter
    def unknown_label(self, value: bool) -> None:
        self.config.DATASET.unknown_label = value

    @property
    def batch_size(self):
        return -1

    @batch_size.setter
    def batch_size(self, value):
        pass

    @property
    def batch_size_timesteps(self):
        return math.floor(self.config.DATASET.batch_size_s / self.resampling_period_s)

    @property
    def train_X_shape(self):
        return (self.num_train_batches, self.batch_size_timesteps, self.num_feature_dims)

    @property
    def train_Y_shape(self):
        return (
            self.num_train_batches,
            self.batch_size_timesteps,
            len(self.label_columns_idxes),
        )

    @property
    def train_actual_batch_len_shape(self):
        return (self.num_train_batches, 1)

    @property
    def train_weights_shape(self):
        return self.train_Y_shape

    @property
    def test_X_shape(self):
        return (self.num_test_batches, self.batch_size_timesteps, self.num_feature_dims)

    @property
    def test_Y_shape(self):
        return (self.num_test_batches, self.batch_size_timesteps, 1)

    @property
    def test_actual_batch_len_shape(self):
        return (self.num_test_batches, 1)

    @property
    def test_weights_shape(self):
        return self.test_Y_shape

    @property
    def verif_X_shape(self):
        return (self.num_verif_batches, self.batch_size_timesteps, self.num_feature_dims)

    @property
    def verif_Y_shape(self):
        return (self.num_verif_batches, self.batch_size_timesteps, 1)

    @property
    def verif_actual_batch_len_shape(self):
        return (self.num_verif_batches, 1)

    @property
    def T_meas_ambient(self):
        return self.experiment.layers.rdp.offset

    def cleanup(self):
        self.remove_dataset()
        for analysis_phase in ["train", "test", "verif"]:
            for run_rep in getattr(self, analysis_phase + "_runs"):
                for filename in run_rep[0].path.rglob("*/*/*" + self.name + "*"):
                    os.remove(filename)

    def write(self):
        self.write_dataset()

    def read(self):
        self.read_dataset()

    @property
    def path(self) -> Path:
        return self.path_dataset

    @property
    def path_dataset(self) -> Path:
        """Get the save path"""
        return self.experiment.path.joinpath(self.name)

    @property
    def num_sensors(self):
        return len(self.config.DATASET.matcher_data["dimensions"])

    @property
    def num_feature_dims(self):
        if not hasattr(self.config.DATASET, "num_feature_dims"):
            self.config.DATASET.num_feature_dims = len(
                self.config.DATASET.matcher_data["dimensions"]
            )
        return self.config.DATASET.num_feature_dims

    def _execute_handler(self, *args, **kwargs):
        """Prepare train and test samples."""
        get_root_logger().info("Start preparation of the train, test and verification samples")
        self.cleanup()
        for phase in ["train", "test", "verif"]:
            if getattr(self, "augmentation_" + phase):
                get_root_logger().info("Using data augmenetation for phase %s" % phase)
                self._prepare_with_augmentation(phase)
            else:
                get_root_logger().info("Not using data augmenetation for phase %s" % phase)
                self._prepare_without_augmentation(phase)

        get_root_logger().info(
            "Finished preparation of the train, test and verification samples"
        )

    @property
    def sample_selection_mode(self):
        if hasattr(self.config.DATASET, "sample_selection"):
            if hasattr(self.config.DATASET.sample_selection, "mode"):
                return SampleSelectionMode(self.config.DATASET.sample_selection.mode)
            else:
                return SampleSelectionMode.Random
        else:
            return SampleSelectionMode.Uniform

    def _init_sample_selection(self, analysis_phase: str = ""):
        self._num_runs = len(getattr(self, analysis_phase + "_runs"))
        self._prev_run = random.randint(0, self._num_runs - 1)
        self._sample_selection = np.zeros((self._num_runs, self._num_runs))
        self._repetition_selection = [
            np.zeros((run_reps[1],)) for run_reps in getattr(self, analysis_phase + "_runs")
        ]
        if self.sample_selection_mode == SampleSelectionMode.Alternating:
            self._sample_lists_selector = True
            self._sample_lists = {True: [], False: []}
            tmp_idx = 0
            for run_rep in getattr(self, analysis_phase + "_runs"):
                if run_rep[0].config.name in self.config.DATASET.sample_selection.runs:
                    self._sample_lists[True].append(tmp_idx)
                else:
                    self._sample_lists[False].append(tmp_idx)
                tmp_idx += 1
            for tmp_key in [True, False]:
                self._sample_lists[tmp_key] = np.array(self._sample_lists[tmp_key])

    def _serialise_sample_selection(self, analysis_phase: str = ""):
        selection_path = self.path.joinpath("sample_selection")
        selection_path.mkdir(exist_ok=True)

        toml_dict = {}
        if self.sample_selection_mode == SampleSelectionMode.Alternating:
            for key in [False, True]:
                np.savetxt(
                    selection_path.joinpath(
                        analysis_phase + "_sample_lists_" + str(key) + ".dat"
                    ),
                    self._sample_lists[key],
                    fmt="%s",
                )
        toml_dict["num_runs"] = self._num_runs
        np.savetxt(
            selection_path.joinpath(analysis_phase + "_sample_selection.dat"),
            self._sample_selection,
            fmt="%d",
        )

        with selection_path.joinpath(analysis_phase + "_sample_selection.toml").open(
            "w"
        ) as tomlfile:
            toml.dump(toml_dict, tomlfile)

    def _choose_and_ingest_run(self, analysis_phase: str = "", run_spec: dict = None):
        if run_spec is not None:
            run_rep = [None, None]
            if "key" in run_spec:
                run_key = run_spec["key"]
                for run in getattr(self, analysis_phase + "_runs"):
                    if run[0].config.name == run_spec["key"]:
                        run_rep[0] = run[0]
            else:
                run_rep[0] = run_spec["run"]
            cur_rep = run_spec["rep"]
        elif self.sample_selection_mode == SampleSelectionMode.Random:
            run_rep = random.choice(getattr(self, analysis_phase + "_runs"))
            cur_rep = random.randint(0, run_rep[1] - 1)
        elif self.sample_selection_mode == SampleSelectionMode.Alternating:
            min_runs = np.where(
                self._sample_selection[
                    self._prev_run, self._sample_lists[self._sample_lists_selector]
                ]
                == self._sample_selection[
                    self._prev_run, self._sample_lists[self._sample_lists_selector]
                ].min()
            )[0]
            min_runs = self._sample_lists[self._sample_lists_selector][min_runs]
            self._sample_lists_selector = not self._sample_lists_selector
        elif self.sample_selection_mode == SampleSelectionMode.Uniform:
            min_runs = np.where(
                self._sample_selection[self._prev_run, :]
                == self._sample_selection[self._prev_run, :].min()
            )[0]

        if (
            self.sample_selection_mode == SampleSelectionMode.Alternating
            or self.sample_selection_mode == SampleSelectionMode.Uniform
        ) and (run_spec is None):
            run_idx = min_runs[self._sample_selection[min_runs, :].sum(axis=1).argmin()]
            self._sample_selection[self._prev_run, run_idx] += 1
            self._prev_run = run_idx
            run_rep = getattr(self, analysis_phase + "_runs")[run_idx]
            cur_rep = random.choice(
                np.where(
                    self._repetition_selection[run_idx]
                    == self._repetition_selection[run_idx].min()
                )[0]
            )

        cur_run = run_rep[0]

        ingest = False
        try:
            cur_run.load_ingestion_data(
                **self.ingest_args(cur_rep, analysis_phase), prefix=self.name + "_"
            )
        except:
            ingest = True

        if ingest:
            get_root_logger().debug(f"No ingestion data present, ingesting run {cur_run}")
            cur_run.ingest(**self.ingest_args(cur_rep, analysis_phase))
            cur_run.save_ingestion_data(prefix=self.name + "_")
        return cur_run

    def _generate_thermal_offset(self, time, T_offset_shape):
        T_offset = np.empty(T_offset_shape)
        dT_ambient = []

        min_duration_ambient = math.ceil((60 * 5) / np.diff(time).mean())
        if self.ambient[0] < self.ambient[1]:
            cur_t_enter = 0

            dT_ambient.append(
                (
                    random.uniform(self.ambient[0], self.ambient[1]) - self.T_meas_ambient,
                    cur_t_enter,
                )
            )
            while cur_t_enter < time.size:
                new_t_enter = random.randint(cur_t_enter + min_duration_ambient, time.size)
                dT_ambient.append(
                    (
                        random.uniform(self.ambient[0], self.ambient[1]) - self.T_meas_ambient,
                        new_t_enter,
                    )
                )
                cur_t_enter = new_t_enter
                if cur_t_enter + min_duration_ambient > time.size:
                    break
        else:
            dT_ambient.append((self.ambient[0], 0))

        for idx in range(len(dT_ambient)):
            idx_enter = dT_ambient[idx][1]
            if idx == 0 and len(dT_ambient) > 1:
                idx_exit = dT_ambient[idx + 1][1]
                T0_z = np.full((1, self.num_sensors), dT_ambient[idx][0])
            elif idx < len(dT_ambient) - 1:
                idx_exit = dT_ambient[idx + 1][1]
                T0_z = T_offset[idx_enter - 1, :].reshape((1, -1))
            else:
                idx_exit = len(time)
                T0_z = T_offset[idx_enter - 1, :].reshape((1, -1))
            T_z = np.full((idx_exit - idx_enter, self.num_sensors), dT_ambient[idx][0])
            T_offset[idx_enter:idx_exit] = self.experiment.layers.lne.augment(
                *[time[idx_enter:idx_exit] - time[idx_enter], T_z, T0_z],
                **{"beta_key": "beta_z_heat" if dT_ambient[idx][0] > 0 else "beta_z_cool"},
            )

        return T_offset + np.random.randint(-1, high=1, size=T_offset.shape, dtype="int32")

    def _prepare_with_augmentation(self, analysis_phase: str):
        """
            generate self.samples and split into training and test data
             0) Init empty stream
             1) Read stream
             2) Resample and normalise stream
             3) Augment and append stream
             4) Repeat 1-3 until sample is finished
             5) Repeat 0-4 until enough self.samples have been initialised
        """

        # --------------------------------------------------------------------------------------------------
        get_root_logger().info(
            "Starting data preparation with augmentation for phase %s" % analysis_phase
        )
        sample_timestamps = np.linspace(
            0,
            (self.batch_size_timesteps - 1) * self.resampling_period_s,
            self.batch_size_timesteps,
        )
        data_col_idx = range(1, min(self.label_columns_idxes))
        self._init_sample_selection(analysis_phase)
        X = np.zeros(getattr(self, analysis_phase + "_X_shape"))
        Y = np.full(
            getattr(self, analysis_phase + "_Y_shape"),
            self.label_mapping["UNKNOWN"]["int"] + 1,
            dtype=np.int32,
        )
        actual_batch_len = np.zeros(getattr(self, analysis_phase + "_actual_batch_len_shape"))

        if analysis_phase == "train" or analysis_phase == "test":
            weights = np.zeros(getattr(self, analysis_phase + "_weights_shape"))
        histlabels = np.zeros((self.num_labels,))
        for batch_idx in range(getattr(self, "num_" + analysis_phase + "_batches")):
            T_0 = None
            remaining_len = self.batch_size_timesteps
            cur_batch_len = 0
            # 1.) Generate the sequence
            while remaining_len >= self.min_len_profile:
                # 1.1.) Choose run and ingest to get the processed thermal and labelling trace
                cur_run = self._choose_and_ingest_run(analysis_phase)
                while cur_run.i_symstream.shape[0] <= 1:
                    cur_run = self._choose_and_ingest_run(analysis_phase)

                # 1.2.) Crop the trace to the required length
                if cur_run.i_symstream.shape[0] > self.min_len_profile:
                    if batch_idx < self.batch_size and analysis_phase != "verif":
                        # This statement should make sure that shorter snippets appear more often
                        max_snippet_len = random.choice(
                            [
                                (1 + batch_idx) * 2 * self.min_len_profile,
                                remaining_len,
                                cur_run.i_symstream.shape[0],
                            ]
                        )
                        min_snippet_len = self.min_len_profile
                        cur_snippet_len = random.randint(
                            min_snippet_len,
                            min([max_snippet_len, remaining_len, cur_run.i_symstream.shape[0]]),
                        )
                    else:
                        # Make sure that at least in the last batch long traces are shown
                        cur_snippet_len = min([remaining_len, cur_run.i_symstream.shape[0]])
                else:
                    cur_snippet_len = cur_run.i_symstream.shape[0]
                context_changes = (
                    np.where(
                        np.diff(cur_run.i_symstream[cur_snippet_len:, self.label_columns_idxes])
                        != 0
                    )[0]
                    + cur_snippet_len
                )

                if len(context_changes) > 0:
                    # Include context change such that the thermal feature of closing the app is still in the thermal trace
                    if cur_snippet_len + self.duration_context_change > remaining_len:
                        cur_snippet_len = remaining_len - self.duration_context_change
                    cur_profile_len = cur_snippet_len + (
                        context_changes[0] + self.duration_context_change - context_changes[0]
                    )
                    trace = np.empty((cur_profile_len, len(self.data_columns_idxes)))
                    trace[:cur_snippet_len, :] = cur_run.i_symstream[
                        :cur_snippet_len, self.data_columns_idxes
                    ]
                    trace[cur_snippet_len:, :] = self.experiment.layers.lne.augment(
                        *[
                            cur_run.i_symstream[
                                context_changes[0] : context_changes[0]
                                + self.duration_context_change,
                                0,
                            ]
                            - cur_run.i_symstream[context_changes[0], 0],
                            cur_run.i_symstream[
                                context_changes[0] : context_changes[0]
                                + self.duration_context_change,
                                self.data_columns_idxes,
                            ],
                            cur_run.i_symstream[cur_snippet_len, self.data_columns_idxes],
                        ]
                    )
                else:
                    cur_profile_len = cur_snippet_len
                    trace = np.empty((cur_profile_len, len(self.data_columns_idxes)))
                    trace[:cur_profile_len, :] = cur_run.i_symstream[
                        :cur_profile_len, self.data_columns_idxes
                    ]

                # 1.3.) Concatenate the traces
                if T_0 is None:
                    T_0 = trace[0, :].reshape((1, -1))
                else:
                    T_0[0, :] = X[batch_idx, cur_batch_len - 1, :]
                Y[
                    batch_idx, cur_batch_len : cur_batch_len + cur_profile_len, :
                ] = cur_run.i_symstream[:cur_profile_len, self.label_columns_idxes]
                X[
                    batch_idx, cur_batch_len : cur_batch_len + cur_profile_len, :
                ] = self.experiment.layers.lne.augment(
                    *[cur_run.i_symstream[:cur_profile_len, 0], trace, T_0]
                )
                remaining_len -= cur_profile_len
                cur_batch_len += cur_profile_len

            # 2. Add augmentation offset to the traces
            sampling_period = np.mean(np.diff(cur_run.i_symstream[:, 0]))
            time = np.linspace(
                0,
                (X[batch_idx, :cur_batch_len, :].shape[0] - 1) * sampling_period,
                X[batch_idx, :cur_batch_len, :].shape[0],
            )
            X[batch_idx, :cur_batch_len, :] += self._generate_thermal_offset(
                time, X[batch_idx, :cur_batch_len, :].shape
            )

            for cur_lbl_col in range(Y.shape[-1]):
                hist, _ = np.histogram(
                    Y[batch_idx, :, cur_lbl_col],
                    bins=np.array(list(range(self.num_labels + 1))) - 0.5,
                    density=False,
                )
                histlabels[:] += hist
            # Set length of each iteration sample
            actual_batch_len[batch_idx, 0] = cur_batch_len
            get_root_logger().info(
                "%s batch %i of %i finished"
                % (
                    analysis_phase,
                    batch_idx,
                    getattr(self, "num_" + analysis_phase + "_batches"),
                )
            )

        if analysis_phase == "train" or analysis_phase == "test":
            per_label_weights = self.compute_weights(
                histlabels, mode=WeightCalculationMode.Proportional
            )
            for label in per_label_weights:
                weights[np.where(Y == label)] = per_label_weights[label]

        Y[Y == self.label_mapping["UNKNOWN"]["int"] + 1] = 0
        setattr(self, analysis_phase + "_X", X)
        setattr(self, analysis_phase + "_Y", Y.astype(int))
        setattr(self, analysis_phase + "_actual_batch_len", actual_batch_len)
        if analysis_phase == "train" or analysis_phase == "test":
            setattr(self, analysis_phase + "_weights", weights)
        self._serialise_sample_selection(analysis_phase)

    # --------------------------------------------------------------------------------------------------
    def _prepare_without_augmentation(self, analysis_phase: str):
        """
        """
        get_root_logger().info(
            "Starting data preparation without augmentation for phase %s" % analysis_phase
        )
        cnt_batch = 0
        if analysis_phase not in ["train", "test", "verif"]:
            raise Exception(f"Unknown analysis phase {analysis_phase}")
        setattr(self.config.DATASET, "num_" + analysis_phase + "_batches", 0)
        reshape_X = list(getattr(self, analysis_phase + "_X_shape"))
        reshape_X[0] = 0
        reshape_Y = list(getattr(self, analysis_phase + "_Y_shape"))
        reshape_Y[0] = 0
        X = np.zeros(reshape_X)
        Y = np.full(reshape_Y, self.label_mapping["UNKNOWN"]["int"] + 1, dtype=np.int32)
        actual_batch_len = list()
        if analysis_phase != "verif":
            weights = np.zeros(reshape_Y)
            histlabels = np.zeros((self.num_labels,))
        for run_rep in getattr(self, analysis_phase + "_runs"):
            for cur_rep in range(run_rep[1]):
                cur_run = self._choose_and_ingest_run(
                    analysis_phase, {"run": run_rep[0], "rep": cur_rep}
                )
                _, num_batches = math.modf(
                    cur_run.i_symstream.shape[0] / self.batch_size_timesteps
                )
                num_batches = int(num_batches) + 1
                for batch_idx in range(num_batches):
                    idx_start = min(
                        [
                            (batch_idx + 0) * self.batch_size_timesteps,
                            cur_run.i_symstream.shape[0],
                        ]
                    )
                    idx_end = min(
                        [
                            (batch_idx + 1) * self.batch_size_timesteps,
                            cur_run.i_symstream.shape[0],
                        ]
                    )
                    x_shape = list(X.shape)
                    x_shape[0] = 1
                    y_shape = list(Y.shape)
                    y_shape[0] = 1
                    X = np.vstack([X, np.zeros(x_shape)])
                    Y = np.vstack([Y, np.zeros(y_shape)])
                    X[-1, : (idx_end - idx_start), :] = (
                        cur_run.i_symstream[idx_start:idx_end, self.data_columns_idxes]
                        + self.T_meas_ambient
                    )
                    Y[-1, : (idx_end - idx_start), :] = cur_run.i_symstream[
                        idx_start:idx_end, self.label_columns_idxes
                    ]

                    # Set length of each iteration sample
                    actual_batch_len.append(
                        cur_run.i_symstream[idx_start:idx_end, self.data_columns_idxes].shape[0]
                    )
                    if analysis_phase != "verif":
                        weights = np.vstack([weights, np.zeros(y_shape)])
                        for cur_lbl_col in range(Y.shape[-1]):
                            hist, _ = np.histogram(
                                Y[batch_idx, :, cur_lbl_col],
                                bins=np.array(list(range(self.num_labels + 1))) - 0.5,
                                density=False,
                            )
                            histlabels[:] += hist
            get_root_logger().debug(
                f"Processing run {cur_run} finished, processed {num_batches}"
            )

        actual_batch_len = np.array(actual_batch_len).reshape((-1, 1))
        if analysis_phase == "verif":
            self.num_verif_batches = X.shape[0]
            setattr(self, "verif_X", X)
            setattr(self, "verif_Y", Y.astype(int))
            setattr(self, "verif_actual_batch_len", actual_batch_len)
        else:
            per_label_weights = self.compute_weights(
                histlabels, mode=WeightCalculationMode.Proportional
            )
            for label in per_label_weights:
                weights[np.where(Y == label)] = per_label_weights[label]
            Y[Y == self.label_mapping["UNKNOWN"]["int"] + 1] = 0
            setattr(self, "num_" + analysis_phase + "_batches", X.shape[0])
            shuffled_idxes = np.array(
                shuffle(
                    list(range(getattr(self, "num_" + analysis_phase + "_batches"))),
                    random_state=random.randint(
                        0, len(getattr(self, analysis_phase + "_runs")) - 1
                    ),
                ),
                dtype=int,
            )
            setattr(self, analysis_phase + "_X", X[shuffled_idxes])
            setattr(self, analysis_phase + "_Y", Y[shuffled_idxes].astype(int))
            setattr(
                self, analysis_phase + "_actual_batch_len", actual_batch_len[shuffled_idxes, :]
            )
            setattr(self, analysis_phase + "_weights", weights[shuffled_idxes])

    def compute_weights(self, histlabels, mode=WeightCalculationMode.Equal):
        """Compute the weights for the labels and return as a dict
        """
        if mode == WeightCalculationMode.Proportional:
            weights = histlabels.sum() / histlabels
            weights[weights == np.inf] = 0
            weights /= weights.max()
            # weights /= np.median(weights)
            return {label: weights[label] for label in range(len(weights))}
        elif mode == WeightCalculationMode.Binary:
            weights = dict()
            for label in self.label_mapping:
                # TODO hackish
                weights[self.label_mapping[label]["int"]] = (
                    1
                    if label != "UNKNOWN" and self.label_mapping[label]["str"] != "HO"
                    else 0.5
                )
            return weights
        elif mode == WeightCalculationMode.Equal:
            return {label: 1.0 for label in range(len(histlabels))}
