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
"""Evaluation analysis of a previously trained recurrent neural network signal decoder."""

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

__all__ = "RNNdecoderEval"


class RNNdecoderEval(RNNdecoder):
    def __init__(self, *args, **kwargs):
        RNNdecoder.__init__(self, *args, **kwargs)

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
        model = load_model(
            self.model_file, custom_objects={"ctc_cost_tensorflow": self.ctc_cost_tensorflow}
        )
        (test_label, tst_pattern) = self.read_data()
        self.label_len = label.shape[1]

        test_pattern = val_pattern
        test_label = val_label

        pred = model.predict(test_pattern)

        y_pred = tf.constant(pred)
        y_true = tf.constant(test_label)
        y_true = tf.squeeze(y_true)

        input_length, y_true = tf.split(y_true, [1, bit + 16], 1)
        label_length = tf.constant(np.full([len(test_pattern), 1], bit + 16, dtype=np.int32))

        cost = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        decode = K.ctc_decode(y_pred, tf.squeeze(input_length))
        sess = tf.Session()
        cost_value = sess.run(cost)
        decoded = sess.run(decode)

        #%%
        decoded_label = np.squeeze(np.array(decoded[0]))[:, : bit + 20]
        groundtruth = np.squeeze(test_label[:, 1:, :])

        identical = np.all(np.equal(decoded_label, groundtruth), 1)

        package_error = 1 - np.sum(identical) / len(test_pattern)

        print("Package error rate is" + str(package_error))
