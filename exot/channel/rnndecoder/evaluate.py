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

__all__ = ("RNNdecoderEval")


import os
import pandas as pd
import numpy as np

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
    # --------------------------------------------------------------------------------------------------
    model     = load_model(self.model_file, custom_objects={'ctc_cost_tensorflow':self.ctc_cost_tensorflow})
    (test_label, tst_pattern)=self.read_data()
    self.label_len = label.shape[1]

    test_pattern=val_pattern
    test_label=val_label

    pred=model.predict(test_pattern)

    y_pred=tf.constant(pred)
    y_true=tf.constant(test_label)
    y_true=tf.squeeze(y_true)

    input_length, y_true=tf.split(y_true,[1,bit+16],1)
    label_length=tf.constant(np.full([len(test_pattern),1],bit+16,dtype=np.int32))

    cost=K.ctc_batch_cost(y_true,y_pred,input_length,label_length)
    decode=K.ctc_decode(y_pred,tf.squeeze(input_length))
    sess=tf.Session()
    cost_value=sess.run(cost)
    decoded=sess.run(decode)

    #%%
    decoded_label=np.squeeze(np.array(decoded[0]))[:,:bit+20]
    groundtruth=np.squeeze(test_label[:,1:,:])

    identical=np.all(np.equal(decoded_label,groundtruth),1)

    package_error=1-np.sum(identical)/len(test_pattern)

    print('Package error rate is'+str(package_error))

