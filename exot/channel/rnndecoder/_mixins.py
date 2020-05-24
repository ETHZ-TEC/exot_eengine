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
"""Mixins for the recurrent neural network signal decoder analysis"""

import os
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf


def cut_minibatch(label, freqv, minibatch_length):
    """Cut minibatches to equal length."""
    cut_length = len(freqv) % minibatch_length
    zip_array = list(zip(label, freqv))
    zip_array.sort(key=lambda x: len(x[1]))
    if cut_length == 0:
        return (label, freqv)
    else:
        del zip_array[-cut_length:]
        return zip(*zip_array)


def tf_count(t, val):
    """Count how many values in a tensor a equal to a value val."""
    cnt_equal_elem = tf.equal(t, val)
    cnt_equal_elem = tf.cast(cnt_equal_elem, tf.int32)
    return tf.reduce_sum(cnt_equal_elem, 1)


def stdr(dataseq):
    """Make data zero mean and with unit standard deviation."""
    flat_dataseq = np.array([y for x in dataseq for y in x])
    dataseq = np.array(dataseq)
    mean = np.mean(flat_dataseq)
    std = np.std(flat_dataseq)
    return (dataseq - mean) / std, mean, std


def stdr_val(val_freqv, mean, std):
    """Make data zero mean and with unit variance with given mean and standard deviation."""
    return (val_freqv - mean) / std


def separate_val(freqv, label):
    """Separate the training and the validation dataset"""
    val_index = np.random.choice(len(freqv), len(freqv) // 10, replace=False)
    val_freqv = freqv[val_index]
    val_label = label[val_index]
    freqv = np.delete(freqv, val_index, 0)
    label = np.delete(label, val_index, 0)
    return freqv, val_freqv, label, val_label, val_index
