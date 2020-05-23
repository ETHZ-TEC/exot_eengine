"""TODO
"""
#from __future__ import absolute_import, division, print_function, unicode_literals
import abc

from datetime import datetime
from functools import partial
from pathlib import Path
import pickle as pk
import time
import toml
import typing as t
import os

import math
import numpy  as np
import pandas as pd
import tensorflow as tf

from keras               import backend as K
from keras               import optimizers, initializers
from keras.models        import Sequential
from keras.layers        import Dense, LSTM, Bidirectional, TimeDistributed
from keras.callbacks     import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence


from exot.channel._base import Analysis
from exot.exceptions import *
from exot.util.attributedict import AttributeDict, LabelMapping
from exot.util.logging import get_root_logger
from exot.util.misc import (
    validate_helper,
)
from exot.util.wrangle import Matcher

from ._mixins import (
    cut_minibatch,
    tf_count,
    stdr,
    stdr_val,
    separate_val
)

__all__ = ("RNNdecoder")


import os
import pandas as pd
import numpy as np

class RNNdecoder(Analysis):
  def __init__(self, *args, **kwargs):
    Analysis.__init__(self, *args, **kwargs)

    validate = partial(validate_helper, self.config, msg="Analysis")

    validate("name", str)
    validate("type", str)

    self.label_length = 0

  @property
  def model_files(self):
    return getattr(self.config.MODEL, 'model_files', None)

  @model_files.setter
  def model_files(self, value: str) -> None:
    setattr(self.config.MODEL, 'model_files', value)

  @property
  def batch_size(self):
    return getattr(self.config.DATASET, 'batch_size', 0)

  @property
  def phases(self):
    return getattr(self.config.DATASET, 'phases', '')

  @property
  def env(self):
    return getattr(self.config.DATASET, 'env', '')

  @property
  def max_len_samples(self):
    return getattr(self.config.DATASET, 'max_len_samples', 0)

  # -------------------------------------------------------------------------------------------- #
  #                                         Overwrites                                           #
  # -------------------------------------------------------------------------------------------- #
  @property
  def path(self) -> Path:
    return self.experiment.path.joinpath(self.name)

  # --------------------------------------------------------------------------------------------------
  # CTC cost function
  def ctc_cost_tensorflow(y_true, y_pred):
    y_true = tf.squeeze(y_true) # Remove dimensions of size 1
    input_length, label_length, y_true = tf.split(y_true,[1,1,self.label_len-2],1)
    return K.ctc_batch_cost(y_true, y_pred, input_length, self.label_length)

  # -------------------------------------------------------------------------------------------- #
  #                                           Methods                                            #
  # -------------------------------------------------------------------------------------------- #
  def read_data():
    label_list = []
    freqv_list = []
    label_unit = np.zeros((0,1), dtype=int)
    freqv_unit = np.zeros((0,1), dtype=np.float64)
    len_cnt = 0
    for run in self.experiment.phases[self.phase]:
      for rep in range(run.config['repetitions']):
        # Assemble path of the files to read from the current run
        pth_label = run.rep_path('Haswell', rep).joinpath('snk.log.csv')
        pth_freqv = run.path('Haswell', rep).joinpath('o_lnestream.dat') # This file was manually generated and would need to be written explicitly if reprocessed.

        # Process the freqv data
        freqv       = pd.read_csv(pth_freqv, sep=' ')
        freqv_value = freqv.values
        freqv_value = freqv_value[:,1]
        if(mode is not None):
          freqv_value = freqv_value-np.min(freqv_value)
          freqv_value = freqv_value/np.max(freqv_value)
        freqv_value = freqv_value[:,np.newaxis]

        if(len(freqv_value) >= self.max_length_samples):
          # Skip logfile if it is too long
          continue

        # Process the label data
        label     = pd.read_csv(pth_label,header=None,names=['label bit'])
        label_row = label.values

        len_cnt  += len(freqv_value)
        if(len_cnt < self.max_length_samples):
          # The total length of the freqv and labels is still below the maximum.
          # Therefore append the full current example to the sample.
          freqv_unit=np.append(freqv_unit, freqv_value, axis=0)
          label_unit=np.append(label_unit, label_row,   axis=0)
          continue
        else:
          # The total length of the freqv exceeds the maximum set, therefore only append
          # a subset of lables.
          label_value      = np.zeros((len(label_unit) // 2 + 2, 1), dtype=int)
          label_value[0,0] = len(freqv_unit)
          label_value[1,0] = len(label_unit) // 2
          cnt = 0
          while(cnt < len(label_unit)):
            # Insert the correct lables for the corresponding frequency scalings
            if(  (label_unit[cnt,0] ==  9) and (label_unit[cnt+1,0] == -2)):
              # Symbol Preamble
              label_value[cnt//2+2,0] = 2
            elif((label_unit[cnt,0] ==  2) and (label_unit[cnt+1,0] == -9)):
              # Symbol Postamble
              label_value[cnt//2+2,0] = 3
            elif((label_unit[cnt,0] ==  2) and (label_unit[cnt+1,0] == -2)):
              # Symbol 0
              label_value[cnt//2+2,0] = 0
            elif((label_unit[cnt,0] == -2) and (label_unit[cnt+1,0] ==  2)):
              # Symbol 1
              label_value[cnt//2+2,0] = 1
            cnt += 2
        # Merge files
        #if len(freqv_value) < self.max_length_samples:
          label_list.append(label_value)
          freqv_list.append(freqv_unit)
          # Reset
          len_cnt    = len(freqv_value)
          freqv_unit = freqv_value
          label_unit = label_row

    return (label_list, freqv_list)

  def read(self):
    pass

  def write(self):
    pass

  def _execute_handler(self, *args, **kwargs):
    pass

