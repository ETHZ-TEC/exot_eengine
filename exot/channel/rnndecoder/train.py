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
"""Training of a recurrent neural network signal decoder."""

import abc
import math
import os
import pickle as pk
import time
import typing as t
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import toml
from keras import backend as K, initializers, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from keras.models import Sequential
from keras.preprocessing import sequence

from exot.channel._base import Analysis
from exot.exceptions import *
from exot.util.attributedict import AttributeDict, LabelMapping
from exot.util.logging import get_root_logger
from exot.util.misc import validate_helper
from exot.util.wrangle import Matcher

from ._mixins import cut_minibatch, separate_val, stdr, stdr_val, tf_count

__all__ = "RNNdecoderTrain"


class RNNdecoderTrain(RNNdecoder):
    def __init__(self, *args, **kwargs):
        RNNdecoder.__init__(self, *args, **kwargs)

        validate(("TRAIN"), AttributeDict)
        validate(("TRAIN", "n_epochs_0"), (int))
        validate(("TRAIN", "n_epochs_1"), (int))
        validate(("TRAIN", "debug_step"), (int))
        validate(("TRAIN", "early_stopping_threshold"), (int))

        validate(("MODEL"), AttributeDict)
        validate(("MODEL", "type"), str)
        validate(("MODEL", "max_grad_norm"), (float, int))
        validate(("MODEL", "p_dropout"), (float, int))
        validate(("MODEL", "num_layers"), (int))
        validate(("MODEL", "num_hidden"), (int))
        validate(("MODEL", "learning_rate"), (float, int))

    @property
    def n_epochs_0(self):
        return getattr(self.config.TRAIN, "n_epochs", 0)

    @property
    def n_epochs_1(self):
        return getattr(self.config.TRAIN, "n_epochs", 0)

    @property
    def start_epoch(self):
        if hasattr(self.config.TRAIN, "start_epoch") and not hasattr(self, "_start_epoch"):
            self._start_epoch = self.config.TRAIN["start_epoch"]
        return getattr(self, "_start_epoch", 0)

    @start_epoch.setter
    def start_epoch(self, value: int) -> None:
        self._start_epoch = value

    @property
    def debug_step(self):
        return getattr(self.config.TRAIN, "debug_step", 0)

    @property
    def early_stopping_threshold(self):
        return getattr(self.config.TRAIN, "early_stopping_threshold", 0)

    @property
    def max_grad_norm(self):
        return getattr(self.config.MODEL, "max_grad_norm", None)

    @property
    def p_dropout(self):
        return getattr(self.config.MODEL, "p_dropout", None)

    @property
    def p_keep(self):
        return 1 - self.model_p_dropout

    @property
    def num_layers(self):
        return getattr(self.config.MODEL, "num_layers", None)

    @property
    def num_hidden(self):
        return getattr(self.config.MODEL, "num_hidden", None)

    @property
    def learning_rate_0(self):
        return getattr(self.config.MODEL, "learning_rate_0", None)

    @property
    def learning_rate_1(self):
        return getattr(self.config.MODEL, "learning_rate_1", None)

    @property
    def decay(self):
        return getattr(self.config.MODEL, "decay", 0)

    @property
    def max_timesteps(self):
        return getattr(self.config.MODEL, "max_timesteps", 0)

    # -------------------------------------------------------------------------------------------- #
    #                                      Overwrites                                              #
    # -------------------------------------------------------------------------------------------- #
    @property
    def path(self) -> Path:
        return self.experiment.path.joinpath(self.name)

    # -------------------------------------------------------------------------------------------- #
    #                                        Methods                                               #
    # -------------------------------------------------------------------------------------------- #

    def read(self):
        pass

    def write(self):
        pass

    def _execute_handler(self, *args, **kwargs):
        # Model implementation, training and validation
        # --------------------------------------------------------------------------------------------------
        # Setup directory for storing the results
        time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_path = "./results/" + time + "_train_Mulpacks_gandalf_norm"
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        print("save_path=" + save_path)

        # --------------------------------------------------------------------------------------------------
        # Prepare the input data
        (label, freqv) = self.read_data()
        print("Total {} samples".format(str(len(freqv))))

        # Convert to numpy
        freqv = np.array(freqv)
        label = np.array(label)

        # --------------------------------------------------------------------------------------------------
        # Separate the training and validation data
        freqv, val_freqv, label, val_label, val_index = separate_val(freqv, label)

        # Standarise the freqv and validation data
        freqv, mean, std = stdr(freqv)
        val_freqv = stdr_val(val_freqv, mean, std)

        # --------------------------------------------------------------------------------------------------
        # Pad the freqv and validation
        freqv = sequence.pad_sequences(
            freqv, maxlen=max_timesteps, dtype="float64", padding="post"
        )
        val_freqv = sequence.pad_sequences(
            val_freqv, maxlen=max_timesteps, dtype="float64", padding="post"
        )
        label = sequence.pad_sequences(label, dtype="int64", padding="post")
        val_label = sequence.pad_sequences(val_label, dtype="int64", padding="post")

        print("label_len=" + str(label.shape[1]) + ", val_label_len=" + str(val_label.shape[1]))
        self.label_len = max(label.shape[1], val_label.shape[1])

        label = sequence.pad_sequences(
            label, maxlen=self.label_len, dtype="int64", padding="post"
        )
        val_label = sequence.pad_sequences(
            val_label, maxlen=self.label_len, dtype="int64", padding="post"
        )
        print(
            "new: label_len="
            + str(label.shape[1])
            + ", val_label_len="
            + str(val_label.shape[1])
        )

        # --------------------------------------------------------------------------------------------------
        # Define the model
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(
                    72,
                    return_sequences=True,
                    implementation=1,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                ),
                input_shape=(max_timesteps, 1),
            )
        )
        model.add(
            Bidirectional(
                LSTM(
                    72,
                    return_sequences=True,
                    implementation=1,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                )
            )
        )
        model.add(
            Bidirectional(
                LSTM(
                    72,
                    return_sequences=True,
                    implementation=1,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                )
            )
        )
        model.add(TimeDistributed(Dense(5, activation="softmax")))
        model.compile(
            optimizer=optimizers.SGD(
                lr=self.learning_rate_0,
                momentum=self.momentum,
                decay=self.decay,
                nesterov=True,
                clipnorm=self.max_grad_norm,
            ),
            loss=self.ctc_cost_tensorflow,
        )
        model.summary()

        # --------------------------------------------------------------------------------------------------
        # Write the config.txt
        config = open(save_path + "/config_info.txt", "w")
        config.write(time + "\n")
        config.write("max_timesteps={}\n".format(max_timesteps))
        config.write("self.batch_size={}\n".format(self.batch_size))
        config.write("Total {} samples\n".format(str(len(label))))
        config.write("self.n_epochs_0={}\n".format(self.n_epochs_0))
        config.close()

        # --------------------------------------------------------------------------------------------------
        # Save the configuration and training/val set
        configuration = (
            time,
            max_timesteps,
            self.batch_size,
            self.n_epochs_0,
            freqv,
            val_freqv,
            label,
            val_label,
            val_index,
            mean,
            std,
        )
        config_tuple = open(save_path + "/config", "wb")
        pickle.dump(configuration, config_tuple)
        config_tuple.close()

        # --------------------------------------------------------------------------------------------------
        # Train the model, log the history
        checkpointer = ModelCheckpoint(
            filepath=save_path + "/Output_checkpoint_best", verbose=1, save_best_only=True
        )
        hist = model.fit(
            freqv,
            label,
            batch_size=self.batch_size,
            epochs=self.n_epochs_0,
            shuffle=True,
            validation_data=(val_freqv, val_label),
            callbacks=[checkpointer],
        )
        history = open(save_path + "/history.txt", "w")
        history.write(str(hist.history))
        print(hist.history)
        history.close()

        # --------------------------------------------------------------------------------------------------
        # Save the model
        model.save(save_path + "/Output_final")

        ####################################################################################################
        # Continue model training
        # --------------------------------------------------------------------------------------------------
        self.label_len = label.shape[1]
        model.compile(
            optimizer=optimizers.SGD(
                lr=self.learning_rate_1,
                momentum=self.momentum,
                decay=self.decay,
                nesterov=True,
                clipnorm=self.max_grad_norm,
            ),
            loss=self.ctc_cost_tensorflow,
        )
        model.summary()
        # --------------------------------------------------------------------------------------------------
        # Check what is the starting score
        score = model.evaluate(val_pattern, val_label, batch_size=self.batch_size)
        print("start with val_loss=" + str(score))

        # --------------------------------------------------------------------------------------------------
        # Create direcotry to save results
        time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_path = "./results/" + time + " <<<< " + load_time
        if not os.isdir(save_path):
            os.mkdir(save_path)
        print("save_path=" + save_path)

        # --------------------------------------------------------------------------------------------------
        # Write the config.txt
        config = open(save_path + "/config_info.txt", "w")
        config.write(time + "\n")
        config.write("max_len={}\n".format(max_len))
        config.write("self.batch_size={}\n".format(self.batch_size))
        config.write("Total {} samples\n".format(str(len(label))))
        config.write("self.n_epochs_1={}\n".format(self.n_epochs_1))
        config.close()

        # --------------------------------------------------------------------------------------------------
        # Save the configuration and training/val set
        configuration = (
            time,
            max_len,
            self.batch_size,
            self.n_epochs_1,
            pattern,
            val_pattern,
            label,
            val_label,
            val_index,
            mean,
            std,
        )
        config_tuple = open(save_path + "/config", "wb")
        pk.dump(configuration, config_tuple)
        config_tuple.close()

        # --------------------------------------------------------------------------------------------------
        # Train the model, log the history
        checkpointer = ModelCheckpoint(
            filepath=save_path + "/Output_checkpoint_best", verbose=1, save_best_only=True
        )
        hist = model.fit(
            pattern,
            label,
            batch_size=self.batch_size,
            epochs=self.n_epochs_1,
            shuffle=True,
            validation_data=(val_pattern, val_label),
            callbacks=[checkpointer],
        )
        history = open(save_path + "/history.txt", "w")
        history.write(str(hist.history))
        print(hist.history)
        history.close()
