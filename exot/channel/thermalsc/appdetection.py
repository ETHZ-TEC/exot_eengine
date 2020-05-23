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
"""TODO
"""
#from __future__ import absolute_import, division, print_function, unicode_literals
import abc

from functools import partial
from pathlib import Path
import pickle as pk
import time
import toml
import typing as t
import os

import editdistance
from fastdtw import fastdtw as FDTW
import math
import numpy  as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import random
import scipy.stats      as st   # Statistical measures
from scipy.spatial.distance import euclidean
from sklearn.utils import shuffle

from scipy import interpolate

from exot.channel._base import Analysis
from exot.channel.mixins.mldataset import (
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
    )
from exot.exceptions import *
from exot.util.attributedict import AttributeDict, LabelMapping
from exot.util.logging import get_root_logger
from exot.util.misc import (
    validate_helper,
)
from exot.util.wrangle import Matcher

from ._mixins import ThermalAppDecectionDataSetHandler

__all__ = ("AppDetection", "ADLSTM", "ADCNNLSTM", "ADConvLSTM", "ADBiDiLSTM")

class AppDetection(
    Analysis,
    ThermalAppDecectionDataSetHandler,
    TrainX, TrainY, TrainSampleLen, TrainWeights,
     TestX,  TestY,  TestSampleLen,  TestWeights,
    VerifX, VerifY, VerifSampleLen,
  ):
  def __init__(self, *args, **kwargs):
    Analysis.__init__(self, *args, **kwargs)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    validate = partial(validate_helper, self.config, msg="Analysis")

    # top-level configuration
    validate("name", str)
    validate("type", str)

    validate(("DATASET"), AttributeDict)
    validate(("DATASET", "load_from"), (str))
    self.read_dataset()

    validate(("TRAIN"), AttributeDict)
    validate(("TRAIN", "n_epochs"), (int))
    validate(("TRAIN", "debug_step"), (int))
    validate(("TRAIN", "early_stopping_threshold"), (int))


    validate(("MODEL"), AttributeDict)
    validate(("MODEL", "type"), str)
    validate(("MODEL", "max_grad_norm"), (float, int))
    validate(("MODEL", "p_dropout"), (float, int))
    validate(("MODEL", "num_layers"), (int))
    validate(("MODEL", "num_hidden"), (int))
    validate(("MODEL", "learning_rate"), (float, int))
    # OPTIONAL validate(("MODEL", "model_files"), (str))

  @property
  def n_epochs(self):
    return getattr(self.config.TRAIN, 'n_epochs', 0)

  @n_epochs.setter
  def n_epochs(self, value):
    self.config.TRAIN.n_epochs = value

  @property
  def start_epoch(self):
    if hasattr(self.config.TRAIN, 'start_epoch') and not hasattr(self, '_start_epoch'):
      self._start_epoch = self.config.TRAIN['start_epoch']
    return getattr(self, '_start_epoch', 0)

  @start_epoch.setter
  def start_epoch(self, value: int) -> None:
    self._start_epoch = value

  @property
  def debug_step(self):
    return getattr(self.config.TRAIN, 'debug_step', 0)

  @property
  def early_stopping_threshold(self):
    return getattr(self.config.TRAIN, 'early_stopping_threshold', 0)

  @property
  def model_type(self):
    return getattr(self.config.MODEL, 'type', None)

  @property
  def max_grad_norm(self):
    return getattr(self.config.MODEL, 'max_grad_norm', None)

  @property
  def p_dropout(self):
    return getattr(self.config.MODEL, 'p_dropout', None)

  @property
  def p_keep(self):
    return 1 - self.model_p_dropout

  @property
  def num_layers(self):
    return getattr(self.config.MODEL, 'num_layers', None)

  @property
  def num_hidden(self):
    return getattr(self.config.MODEL, 'num_hidden', None)

  @property
  def learning_rate(self):
    return getattr(self.config.MODEL, 'learning_rate', None)

  @property
  def model_files(self):
    return getattr(self.config.MODEL, 'model_files', None)

  @model_files.setter
  def model_files(self, value: str) -> None:
    setattr(self.config.MODEL, 'model_files', value)

  @property
  def filtering_window_size(self):
    return getattr(self.config.MODEL, 'filtering_window_size', 25)

  @property
  def unknown_label(self):
    return getattr(self.config.DATASET, 'unknown_label', False)

  @unknown_label.setter
  def unknown_label(self, value: bool) -> None:
    self.config.DATASET.unknown_label = value

  @property
  def max_timesteps(self):
    return getattr(self.config.MODEL, 'max_timesteps', 0)

  @property
  def batch_size(self):
    if not hasattr(self, '_batch_size'):
      self._batch_size = max([
          math.ceil(self.train_actual_batch_len.max() / self.max_timesteps),
          math.ceil(self.test_actual_batch_len.max() / self.max_timesteps),
          math.ceil(self.verif_actual_batch_len.max() / self.max_timesteps),
        ])
    return self._batch_size

  @batch_size.setter
  def batch_size(self, value):
    pass

  def reshuffle_data(self, phase: str, epoch: int) -> None:
    X, Y, actual_batch_len, weights = shuffle(
        getattr(self, phase + '_X'),
        getattr(self, phase + '_Y'),
        getattr(self, phase + '_actual_batch_len'),
        getattr(self, phase + '_weights'),
      )
    setattr(self, phase + '_X', X)
    setattr(self, phase + '_Y', Y)
    setattr(self, phase + '_actual_batch_len', actual_batch_len)
    setattr(self, phase + '_weights',  weights)

  @property
  def train_X_shape(self):
    return (self.batch_size, self.max_timesteps, self.num_feature_dims)

  @property
  def train_Y_shape(self):
    return (self.batch_size, self.max_timesteps)

  @property
  def train_actual_batch_len_shape(self):
    return (self.batch_size)

  @property
  def train_weights_shape(self):
    return self.train_Y_shape

  @property
  def test_X_shape(self):
    return self.train_X_shape

  @property
  def test_Y_shape(self):
    return self.train_Y_shape

  @property
  def test_actual_batch_len_shape(self):
    return (self.batch_size)

  @property
  def test_weights_shape(self):
    return self.test_Y_shape

  @property
  def verif_X_shape(self):
    return self.train_X_shape

  @property
  def verif_Y_shape(self):
    return train_Y_shape

  @property
  def verif_actual_batch_len_shape(self):
    return (self.batch_size)

  def get_data(self, phase: str, batch: int) -> tuple:
    X = np.zeros([self.batch_size, self.max_timesteps, self.num_feature_dims])
    Y = np.zeros([self.batch_size, self.max_timesteps])
    w = np.ones([self.batch_size, self.max_timesteps])

    full_samples, last_sample_len = divmod(getattr(self, phase + '_actual_batch_len')[batch, 0], self.max_timesteps)
    #full_samples, last_sample_len = divmod(getattr(self, phase + '_X').shape[1], self.max_timesteps)
    full_samples = int(full_samples); last_sample_len = int(last_sample_len)

    if phase == 'verif':
      if self.batch_size_timesteps > getattr(self, phase + '_X').shape[1]:
        X[:full_samples, :, :]               = getattr(self, phase + '_X')[batch, :full_samples*self.max_timesteps, :].reshape([full_samples, self.max_timesteps, self.num_feature_dims])
        X[full_samples, :last_sample_len, :] = getattr(self, phase + '_X')[batch, full_samples*self.max_timesteps:, :].reshape([last_sample_len, self.num_feature_dims])

        Y[:full_samples, :]                  = getattr(self, phase + '_Y')[batch, :full_samples*self.max_timesteps, :].squeeze().reshape([full_samples, self.max_timesteps])
        Y[full_samples, :last_sample_len]    = getattr(self, phase + '_Y')[batch, full_samples*self.max_timesteps:, :].squeeze().reshape([last_sample_len])

        w[full_samples, :last_sample_len]    = 0
      else:
        X = getattr(self, phase + '_X')[batch, :self.batch_size_timesteps, :].reshape([self.batch_size, self.max_timesteps, self.num_feature_dims])
        Y = getattr(self, phase + '_Y')[batch, :self.batch_size_timesteps, :].squeeze().reshape([self.batch_size, self.max_timesteps])
    else:
      cur_position = 0
      for sample_idx in range(self.batch_size):
        if sample_idx == self.batch_size - 1:
          # For training, make sure the last sample has between 1% and 10% of zero padding at the end
          actual_sample_len = random.randint(math.floor(0.9*self.max_timesteps), math.ceil(0.95*self.max_timesteps))
        else:
          actual_sample_len = self.max_timesteps
        X[sample_idx, :actual_sample_len, :] = getattr(self, phase + '_X')[batch, cur_position:cur_position+actual_sample_len, :].reshape([actual_sample_len, self.num_feature_dims])
        Y[sample_idx, :actual_sample_len]    = getattr(self, phase + '_Y')[batch, cur_position:cur_position+actual_sample_len, :].reshape([actual_sample_len])
        w[sample_idx, :actual_sample_len]    = getattr(self, phase + '_weights')[batch, cur_position:cur_position+actual_sample_len, :].reshape([actual_sample_len])
        w[sample_idx, actual_sample_len:]    = 0

        cur_position += actual_sample_len

    return X.astype(np.float32), Y.astype(int), w.astype(np.float32)

  # -------------------------------------------------------------------------------------------- #
  #                                      Overwrites                                              #
  # -------------------------------------------------------------------------------------------- #
  @property
  def path(self) -> Path:
    return self.experiment.path.joinpath(self.name)

  @property
  def path_dataset(self) -> Path:
    """Get the save path"""
    return self.experiment.path.joinpath(self.config.DATASET.load_from)

  # -------------------------------------------------------------------------------------------- #
  #                       Methods                      #
  # -------------------------------------------------------------------------------------------- #
  def read(self):
    pass

  def write(self):
    pass

  def _execute_handler(self, *args, **kwargs):
    self.init_app_detector()
    best_model = self.train()
    self.verify(best_model)

  def write_debug_files(self, phase: str, outputs=None, prefix='', as_txt=False):
    """Serialise a checkpoint"""
    debug_path = self.path.joinpath(phase)
    debug_path.mkdir(exist_ok=True)
    #Take advantage of the writing methods
    if outputs is not None:
      for name, value in outputs.items():
        if isinstance(value, np.ndarray):
          if as_txt:
            np.savetxt(debug_path / (prefix + '_' + name + ".dat"), value, fmt='%.3f')
          else:
            self._save_npz(debug_path / (prefix + '_' + name + ".npz"), {name: value})
        elif isinstance(value, (pd.DataFrame, pd.Series, pd.Panel)):
          self._save_hdf(debug_path / (prefix + '_' + name + ".h5"), {name: value})
        elif isinstance(value, (dict, AttributeDict)):
          with debug_path.joinpath(prefix + '_' + name + ".toml").open("w") as toml_file:
            toml.dump(value, toml_file)
        else:
          with open(debug_path.joinpath(prefix + '_' + name + '.pkl'), 'wb') as f:
            pk.dump(value, f)

  def read_debug_files(self, phase: str, prefix=''):
    """Load a checkpoint"""
    debug_path = self.path.joinpath(phase)
    #Take advantage of the writing methods
    loaded_data = AttributeDict()
    for _data_file_path in os.listdir(debug_path):
      _data_file_path = debug_path.joinpath(_data_file_path)
      if _data_file_path.name.startswith(prefix):
        _fileextension = _data_file_path.name.split('.')[-1]
        key            = _data_file_path.name.replace(prefix + '_', '')
        key            = key.replace('.' + _fileextension, '')
        if _fileextension == 'npz':
          with np.load(_data_file_path) as _np_store:
            for data_key in _np_store:
              loaded_data[data_key] = _np_store[data_key]
        elif _fileextension == 'dat':
          loaded_data[key] = np.loadtxt(_data_file_path)
        elif _fileextension == 'h5':
          with pd.HDFStore(_data_file_path, "r") as _pd_store:
            for data_key in _pd_store:
              loaded_data[data_key] = _pd_store[data_key]
        elif _fileextension == 'toml':
          with _data_file_path.open("r") as toml_file:
            loaded_data[key] = toml.load(toml_file)
        elif _fileextension == 'pkl':
          with open(_data_file_path, 'rb') as f:
            loaded_data[key] = pk.load(f)
        else:
            raise Exception(f"Unknown file extension for file {_data_file_path}")
    return loaded_data

  def init_app_detector(self):
    appdetector_args = {
                         'batch_size':self.batch_size,
                         'num_hidden':self.num_hidden,
                         'num_layers':self.num_layers,
                         'num_labels':self.num_labels,
                         'p_dropout':self.p_dropout,
                         'max_timesteps':self.max_timesteps,
                         'num_features':self.max_timesteps,
                       }
    if self.model_type == 'LSTM':
      self.appdetector = ADLSTM(**appdetector_args)
    elif self.model_type == 'CNNLSTM':
      self.appdetector = ADCNNLSTM(**appdetector_args)
    else:
      raise Exception(NotImplemented)
    self.optimizer = tf.keras.optimizers.RMSprop(
                                                 learning_rate=self.learning_rate,
                                                 rho=0.9,
                                                 momentum=0.1,
                                                 epsilon=1e-07,
                                                 centered=False,
                                                 name='RMSprop',
                                                )
    self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                          appdetector=self.appdetector)

  def train(self):
    get_root_logger().info("Epoch X | loss: (TRAIN/TEST) | accuracy: (TRAIN/TEST) | lev_abs: (TRAIN/TEST) | lev_len: (TRAIN/TEST) | lev_norm: (TRAIN/TEST) | timing: (TRAIN/TEST)")
    self.best_lev_norm = np.inf
    self.best_loss     = np.inf
    best_model         = None
    early_stop         = False

    trace_loss       = {'train':np.empty((math.ceil(self.n_epochs / self.debug_step))), 'test':np.empty((math.ceil(self.n_epochs / self.debug_step)))}
    trace_accuracy   = {'train':np.empty((math.ceil(self.n_epochs / self.debug_step))), 'test':np.empty((math.ceil(self.n_epochs / self.debug_step)))}
    trace_lev_abs    = {'train':np.empty((math.ceil(self.n_epochs / self.debug_step))), 'test':np.empty((math.ceil(self.n_epochs / self.debug_step)))}
    trace_lev_norm   = {'train':np.empty((math.ceil(self.n_epochs / self.debug_step))), 'test':np.empty((math.ceil(self.n_epochs / self.debug_step)))}
    trace_lev_len    = {'train':np.empty((math.ceil(self.n_epochs / self.debug_step))), 'test':np.empty((math.ceil(self.n_epochs / self.debug_step)))}
    trace_timing     = {'train':np.empty((math.ceil(self.n_epochs / self.debug_step))), 'test':np.empty((math.ceil(self.n_epochs / self.debug_step)))}

    loss       = {'train':None, 'test':None}
    accuracy   = {'train':None, 'test':None}
    lev_abs    = {'train':None, 'test':None}
    lev_norm   = {'train':None, 'test':None}
    lev_len    = {'train':None, 'test':None}
    timing     = {'train':None, 'test':None}
    pred       = {'train':None, 'test':None}
    Y          = {'train':None, 'test':None}
    X          = {'train':None, 'test':None}
    weights    = {'train':None, 'test':None}
    dbg_cnt    = 0
    for epoch in range(self.start_epoch+1, self.n_epochs+1):
      appdetector_state = self.appdetector.initial_hidden_state
      # Training
      self.reshuffle_data('train', epoch)
      debug = ((epoch % self.debug_step) == 0 or epoch == self.n_epochs or epoch == 1)
      (pred['train'], Y['train'], X['train'], weights['train'],
       loss['train'], accuracy['train'],
       lev_abs['train'], lev_norm['train'], lev_len['train'],
       timing['train']) = self.inference(epoch, train=True, debug=debug)
      if((epoch % self.debug_step) == 0 or epoch == self.n_epochs or epoch == 1):
        (pred['test'], Y['test'], X['test'], weights['test'],
        loss['test'], accuracy['test'],
        lev_abs['test'], lev_norm['test'], lev_len['test'],
        timing['test']) = self.inference(epoch, train=False, debug=debug) # Test inference

        for phase in ['train', 'test']:
          trace_loss[phase][dbg_cnt]     = np.nansum(loss[phase])
          trace_accuracy[phase][dbg_cnt] = np.nanmean(accuracy[phase])
          trace_lev_abs[phase][dbg_cnt]  = np.nansum(lev_abs[phase])
          trace_lev_norm[phase][dbg_cnt] = np.nanmean(lev_norm[phase])
          trace_lev_len[phase][dbg_cnt]  = np.nansum(lev_len[phase])
          trace_timing[phase][dbg_cnt]   = np.nanmean(timing[phase])

        get_root_logger().info(
          "Epoch %03i | loss: (%03.3f/%03.3f) | accuracy: (%2.2f%%/%2.2f%%) | lev_abs: (%4i/%4i) | lev_len: (%4i/%4i) | lev_norm: (%.3f/%.3f) | timing: (%.3f/%.3f)" %
            (epoch,
             trace_loss['train'][dbg_cnt]    ,   trace_loss['test'][dbg_cnt],
             trace_accuracy['train'][dbg_cnt],   trace_accuracy['test'][dbg_cnt],
             trace_lev_abs['train'][dbg_cnt] ,   trace_lev_abs['test'][dbg_cnt],
             trace_lev_len['train'][dbg_cnt] ,   trace_lev_len['test'][dbg_cnt],
             trace_lev_norm['train'][dbg_cnt],   trace_lev_norm['test'][dbg_cnt],
             trace_timing['train'][dbg_cnt]  ,   trace_timing['test'][dbg_cnt],
            )
        )

        #if trace_lev_norm['test'][dbg_cnt] < self.best_lev_norm:
        if trace_loss['test'][dbg_cnt] == 0.0:
          train_violations = self.early_stopping_threshold + 1
        if trace_loss['test'][dbg_cnt] < self.best_loss:
          train_violations = 0
          #self.best_lev_norm = trace_lev_norm['test'][dbg_cnt]
          self.best_loss = trace_loss['test'][dbg_cnt]
          for phase in ['train', 'test']:
            self.write_debug_files(phase=phase, outputs=pred[phase], prefix='prediction')
            self.write_debug_files(phase=phase, outputs=Y[phase],    prefix='Y')
            self.write_debug_files(phase=phase, outputs={'X':X[phase],'weights':weights[phase]}, prefix='data')
            self.write_debug_files(phase=phase, outputs={
                                                          'accuracy':accuracy[phase],
                                                          'lev_abs':lev_abs[phase],
                                                          'lev_norm':lev_norm[phase],
                                                          'lev_len':lev_len[phase],
                                                          'timing':timing[phase],
                                                          'results':{
                                                                      'epoch':epoch,
                                                                      'loss':trace_loss[phase][dbg_cnt],
                                                                      'accuracy':trace_accuracy[phase][dbg_cnt],
                                                                      'lev_abs':trace_lev_abs[phase][dbg_cnt],
                                                                      'lev_norm':trace_lev_norm[phase][dbg_cnt],
                                                                      'timing':trace_timing[phase][dbg_cnt],
                                                                    },
                                                        }, prefix='metric', as_txt=True)
          best_model = self.checkpoint.save(file_prefix=self.path.joinpath("tf_ckpt"))
          self.config.MODEL['best_model'] = best_model
          with self.path.joinpath("model.toml").open("w") as toml_file:
            toml.dump(self.config.MODEL, toml_file)
        else:
          train_violations += self.debug_step if epoch >= self.debug_step else 1
        dbg_cnt += 1
      if train_violations >= self.early_stopping_threshold:
        if not early_stop:
          new_lr = self.learning_rate / 10
          get_root_logger().info("Starting second phase of training at %i epochs with learning rate %f!" % (epoch, new_lr))
          get_root_logger().info("Restoring previously best model from %s!" % str(best_model))
          self.checkpoint.restore(best_model)
          config = self.optimizer.get_config()
          config['learning_rate'] = self.learning_rate / 10
          self.optimizer = self.optimizer.from_config(config)
          early_stop = True
          train_violations = 0
        else:
          get_root_logger().info("Early stopping triggered after %i epochs!" % epoch)
          break

    for phase in ['train', 'test']:
      self.write_debug_files(phase=phase, outputs={
                                                'loss'    : trace_loss[phase],
                                                'accuracy': trace_accuracy[phase],
                                                'lev_abs' : trace_lev_abs[phase],
                                                'lev_norm': trace_lev_norm[phase],
                                                'lev_len' : trace_lev_len[phase],
                                                'timing'  : trace_timing[phase],
                                               }, prefix='trace', as_txt=True)
    get_root_logger().info("Training finished, saved files to " + str(self.path.joinpath('train')) + " and " + str(self.path.joinpath('test')))
    return best_model

  def debug_restore_checkpoint(self, phase: str = 'train'):
    pred     = self.read_debug_files(phase=phase, prefix='prediction')
    Y        = self.read_debug_files(phase=phase, prefix='Y')
    data     = self.read_debug_files(phase=phase, prefix='data')
    metrics  = self.read_debug_files(phase=phase, prefix='metric')
    traces   = self.read_debug_files(phase=phase, prefix='trace')
    return pred, Y, data, metrics, traces


  def verify(self, best_model = None):
    if best_model is not None:
      self.checkpoint.restore(best_model)
      get_root_logger().info("Restored model from " + str(best_model))
    if self.verif_X is None:
      get_root_logger().info("No Inference data, nothing to do...")
    else:
      (pred, Y, X, weights, loss, accuracy, lev_abs, lev_norm, lev_len, timing) = self.inference(train=False, epoch=None, debug=True)
      overall_loss     = np.nansum(loss)
      overall_accuracy = np.nanmean(accuracy)
      overall_lev_abs  = np.nansum(lev_abs)
      overall_lev_norm = np.nanmean(lev_norm)
      overall_lev_len  = np.nansum(lev_len)
      overall_timing   = np.nanmean(timing)
      get_root_logger().info(
        "Verification | loss: %03.3f | accuracy: %2.2f%% | lev_abs: %4i | lev_len: %4i | lev_norm: %.3f | timing: %.3f" %
          (overall_loss, overall_accuracy, overall_lev_abs, overall_lev_len, overall_lev_norm, overall_timing,)
      )
      self.write_debug_files(phase='verif', outputs=pred, prefix='prediction')
      self.write_debug_files(phase='verif', outputs=Y,    prefix='Y')
      self.write_debug_files(phase='verif', outputs=self.test_set, prefix='')
      self.write_debug_files(phase='verif', outputs={'X':X,'weights':weights}, prefix='data')
      self.write_debug_files(phase='verif', outputs={
                                                    'accuracy':accuracy,
                                                    'lev_abs':lev_abs,
                                                    'lev_norm':lev_norm,
                                                    'lev_len':lev_len,
                                                    'timing':timing,
                                                    'results':{
                                                                'loss':overall_loss,
                                                                'accuracy':overall_accuracy,
                                                                'lev_abs':overall_lev_abs,
                                                                'lev_norm':overall_lev_norm,
                                                                'timing':overall_timing,
                                                              },
                                                  }, prefix='metric', as_txt=True)
      self.checkpoint.save(file_prefix=self.path.joinpath('verif').joinpath("tf_ckpt"))
      get_root_logger().info("Inference finished, saved files to " + str(self.path.joinpath('verif')))

  def inference(self, epoch: int = None, train: bool = True, debug: bool = True):
    if train:
      phase = 'train'
    else:
      phase = 'verif' if epoch is None else 'test'
    num_batches = getattr(self, 'num_' + phase + '_batches')
    pred    = {
               'raw'     :[None for _ in range(num_batches)],
               'filtered':[None for _ in range(num_batches)],
               'collapse':[None for _ in range(num_batches)],
               'timingCX':[None for _ in range(num_batches)],
              }
    Y       = {
               'raw'     :[None for _ in range(num_batches)],
               'collapse':[None for _ in range(num_batches)],
               'timingCX':[None for _ in range(num_batches)],
              }
    X       = [None for _ in range(num_batches)]
    weights = [None for _ in range(num_batches)]

    loss     = np.zeros((num_batches))
    accuracy = np.zeros((num_batches))
    lev_abs  = np.zeros((num_batches))
    lev_norm = np.zeros((num_batches))
    lev_len  = np.zeros((num_batches))
    timing   = np.zeros((num_batches))

    if epoch is not None:
      self.reshuffle_data(phase, epoch)
    appdetector_state = self.appdetector.initial_hidden_state
    for batch in range(num_batches):
      X[batch], Y['raw'][batch], weights[batch] = self.get_data(phase, batch)

      # Use application detector to get per timestep predictions
      loss[batch], tf_pred, appdetector_state = self.process(X[batch],
                                                             Y['raw'][batch][:,0::self.appdetector.oversampling_rate],
                                                             weights[batch][:,0::self.appdetector.oversampling_rate],
                                                             appdetector_state,
                                                             train=train)
      if debug:
        pred['raw'][batch] = tf_pred.numpy()

        if pred['raw'][batch].size != Y['raw'][batch].size:
          timesteps_new = np.arange(0, Y['raw'][batch].shape[1])
          timesteps_old = np.linspace(0, Y['raw'][batch].shape[1],pred['raw'][batch].shape[1])
          pred_resampled = np.empty(Y['raw'][batch].shape)
          for sample_idx in range(Y['raw'][batch].shape[0]):
            label_interpolation          = interpolate.interp1d(timesteps_old, pred['raw'][batch][sample_idx,:], kind='nearest')
            pred_resampled[sample_idx,:] = label_interpolation(timesteps_new)
          pred['raw'][batch] = pred_resampled

        # Apply majority voting to compensate for jittery predictions
        pred['filtered'][batch] = self.filter_batch(pred['raw'][batch], window_size=self.filtering_window_size)

        # Collapse labels such that only the sequence without duplicates remains
        pred['collapse'][batch], pred['timingCX'][batch] = self.get_sequences(pred['filtered'][batch], weights[batch])
        Y['collapse'][batch],    Y['timingCX'][batch]    = self.get_sequences(Y['raw'][batch], weights[batch])

        # Calculate metrics
        accuracy[batch]                                 = self.batch_accuracy(pred['filtered'][batch], Y['raw'][batch], weights[batch])
        lev_abs[batch], lev_norm[batch], lev_len[batch] = self.batch_levensthein_distance(pred['collapse'][batch], Y['collapse'][batch])
        timing[batch]                                   = self.batch_timing_accuracy(Y['timingCX'][batch], pred['timingCX'][batch])
    #return np.nansum(loss), np.nanmean(accuracy), np.nansum(lev_abs), np.nanmean(lev_norm), np.nansum(lev_len), np.nanmean(timing)
    return pred, Y, X, weights, loss, accuracy, lev_abs, lev_norm, lev_len, timing

  @tf.function
  def process(self, inp, targ=None, weights=None, appdetector_state=None, train=True):
    loss       = 0
    batch_loss = 0

    with tf.GradientTape() as tape:
      enc_output, appdetector_state = self.appdetector(inp, appdetector_state)
      if targ is not None and weights is not None:
        loss = tf.reduce_sum(tfa.seq2seq.sequence_loss(
           enc_output,
           targ,
           weights,
           average_across_timesteps=False,
           average_across_batch=True,
           sum_over_timesteps=False,
           sum_over_batch=False,
           softmax_loss_function=None,
           name='Seq2SeqLoss'
           ))
    if targ is not None:
      batch_loss = (loss / int(targ.shape[1]))

    if train:
      # Train with gradient clipping
      gradients         = tape.gradient(loss, self.appdetector.trainable_variables)
      gradients_no_nans = [tf.where(tf.math.is_nan(x), tf.zeros_like(x), x) for x in gradients]
      gradients, _      = tf.clip_by_global_norm(gradients, self.max_grad_norm)
      self.optimizer.apply_gradients(zip(gradients, self.appdetector.trainable_variables))

    predictions = tf.argmax(enc_output, 2, output_type=tf.int32)

    return batch_loss, predictions, appdetector_state

  # --------------------------------------------------------------------------------------------------
  # Sliding window over the data. Majority voting is performed to replace outliers
  # and reduce noise.
  def filter_batch(self, data, window_size=25):
    filtered = np.empty((data.shape[0],data.shape[1]))
    for iteration in range(data.shape[0]):
      filtered[iteration] = self.majority_voting(data[iteration], window_size)
    return filtered

  def majority_voting(self, data, window_size=25):
    if window_size == 0:
      out_data = data.copy()
    else:
      out_data = np.empty(data.shape)

      if len(data.shape) == 1:
        def majority_filter(idx):
          out_data[idx] = st.mode(data[max([0, math.floor(idx-window_size/2)]):min([math.ceil(idx+window_size/2), data.shape[0]])], axis=0)[0]
      else:
        def majority_filter(idx):
          out_data[idx, :] = st.mode(data[max([0, math.floor(idx-window_size/2)]):min([math.ceil(idx+window_size/2), data.shape[0]]), :], axis=0)[0]

      filtering_function = np.vectorize(majority_filter)
      filtering_function(np.arange(0, data.shape[0]))
    return out_data

  def get_sequences(self, labels, weights=None):
    sequences = [None for _ in range(labels.shape[0])]
    timings   = [None for _ in range(labels.shape[0])]
    for iteration in range(labels.shape[0]):
      tf_labels_collapsed, tf_labels_change, tf_labels_masked = self.tf_collapse_labels(labels[iteration,:], weights[iteration,:])
      sequences[iteration] = tf_labels_collapsed.numpy()
      timings[iteration]   = tf_labels_change.numpy()
      if tf_labels_masked.numpy().size > 0:
        sequences[iteration] = np.concatenate([sequences[iteration], np.array([tf_labels_masked.numpy()[0]])])
    return sequences, timings

  @tf.function
  def tf_collapse_labels(self, labels, weights=None):
    if weights is not None:
      labels_masked = tf.cast(tf.boolean_mask(labels, tf.greater(weights, tf.constant(0, dtype=tf.float32))), tf.int32)
    else:
      labels_masked = tf.cast(labels, tf.int32)

    labels_diff      = tf.subtract(labels_masked[:-1], labels_masked[1:])
    labels_change    = tf.where(labels_diff != tf.constant(0, dtype=tf.int32))
    labels_collapsed = tf.boolean_mask(labels_masked, tf.not_equal(labels_diff, tf.constant(0, dtype=tf.int32)))

    return labels_collapsed, labels_change, labels_masked

  def batch_accuracy(self, predictions, target, weights=None):
    accuracy = np.empty((target.shape[0]))
    if weights is not None:
      for iteration in range(target.shape[0]):
        accuracy[iteration] = self.tf_accuracy(predictions[weights > 0], target[weights > 0])
    else:
      for iteration in range(target.shape[0]):
        accuracy[iteration] = self.tf_accuracy(predictions, target)
    return np.nanmean(accuracy) * 100

  @tf.function
  def tf_accuracy(self, predictions, target):
    return tf.divide(tf.size(tf.boolean_mask(predictions, tf.equal(tf.cast(predictions, tf.int32), tf.cast(target, tf.int32)))), tf.size(target))

  def batch_levensthein_distance(self, predictions, target):
    lev_abs  = np.empty((len(target)))
    lev_norm = np.empty((len(target)))
    lev_len  = np.empty((len(target)))
    for iteration in range(len(target)):
      lev_abs[iteration]  = editdistance.eval(predictions[iteration], target[iteration])
      lev_norm[iteration] = lev_abs[iteration] / target[iteration].size
      lev_len[iteration]  = target[iteration].size
    return np.nansum(lev_abs), np.nanmean(lev_norm), np.nansum(lev_len)

  def batch_timing_accuracy(self, predictions, target):
    timings = np.empty((len(target)))
    for iteration in range(len(target)):
      timings[iteration], _ = FDTW(target[iteration], predictions[iteration], dist=euclidean)
    return timings.mean()

class AppDetector(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super(AppDetector, self).__init__()
    self._init_handler(*args, **kwargs)

  @property
  def oversampling_rate(self):
    return 1

  @abc.abstractmethod
  def _init_handler(self, *args, **kwargs):
    pass

  @abc.abstractmethod
  def call(self, x, state):
    pass

  @property
  def initial_hidden_state(self):
    return ([tf.zeros(self.state_size) for _ in range(self.num_layers)], [tf.zeros(self.state_size) for _ in range(self.num_layers)])

  @property
  def state_size(self):
    return (self.batch_sz, self.enc_units)

class ADBiDiLSTM(AppDetector):
  pass

class ADConvLSTM(AppDetector):
  pass

class ADCNNLSTM(AppDetector):
  @property
  def oversampling_rate(self):
    return np.prod(self.kernel_size)

  def _init_handler(self, *args, **kwargs):
    self.batch_sz      = kwargs['batch_size']
    self.enc_units     = kwargs['num_hidden']
    self.num_layers    = kwargs['num_layers']
    self.num_labels    = kwargs['num_labels']
    self.max_timesteps = kwargs['max_timesteps']


    # TODO this is hardcoded for now, could be included into the configuration file
    # TODO then self.num_layers should be renamed to self.num_lstm_layers
    self.num_cnn_layers = 1
    filters             = [64]        #np.linspace(8,256,self.num_cnn_layers)
    self.kernel_size    = [25]        # Results in an output samping step of 1 Seconds
    #
    # Possible working configuration:
    # 1st CNN layer kernel size 25 (1s)
    # 2nd CNN layer kernel size 10
    #self.num_cnn_layers = 2
    #filters             = [64, 256]   #np.linspace(8,256,self.num_cnn_layers)
    #self.kernel_size    = [25, 10]     # Results in an output samping step of 1 Seconds

    self.cnn = [tf.keras.layers.Conv1D(
        filters=filters[layer_idx],
        kernel_size=self.kernel_size[layer_idx],
        activation='relu',
        input_shape=(self.batch_sz, int(self.max_timesteps / np.prod(self.kernel_size[:layer_idx+1])), None),
        padding='causal',
        kernel_initializer='he_uniform',
        data_format="channels_last"
        ) for layer_idx in range(self.num_cnn_layers)]
    self.pooling = [tf.keras.layers.MaxPool1D(
        pool_size=self.kernel_size[layer_idx],
        padding="same",
        data_format="channels_last",
        input_shape=(self.batch_sz, int(self.max_timesteps / np.prod(self.kernel_size[:layer_idx+1])), None)
        ) for layer_idx in range(self.num_cnn_layers)]

    #self.masking = tf.keras.layers.Masking(mask_value=np.nan)
    self.lstm = [tf.keras.layers.LSTM(
                                      self.enc_units,
                                      activation='tanh',
                                      #use_bias=,
                                      kernel_initializer='glorot_uniform',
                                      recurrent_initializer='glorot_uniform',
                                      #kernel_initializer='he_uniform',
                                      #recurrent_initializer='he_uniform',
                                      bias_initializer='zeros',
                                      #unit_forget_bias=,
                                      #kernel_regularizer=,
                                      #recurrent_regularizer=,
                                      #bias_regularizer=,
                                      #activity_regularizer=,
                                      #kernel_constraint=,
                                      #bias_constraint=,
                                      dropout=kwargs['p_dropout'],
                                      #recurrent_dropout=kwargs['p_dropout'],
                                      #implementation=2,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_dropout=0,
                                      go_backwards=False,
                                      stateful=True,
                                      unroll=False
                                     ) for _ in range(self.num_layers)]
    self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
    #self.dense = tf.keras.layers.Dense(
                                       self.num_labels,
                                       #activation='relu',
                                       activation='linear',
                                       #use_bias=,
                                       #kernel_initializer='he_uniform',
                                       kernel_initializer='glorot_uniform',
                                       bias_initializer='zeros',
                                       #kernel_regulariz
                                       #bias_regularizer=,
                                       #activity_regularizer=,
                                       #kernel_constraint=,
                                       #bias_constraint=,
                                       ),
                                       input_shape=(None,kwargs['max_timesteps'],self.enc_units)
                                      )

  def call(self, x, state):
    output = x
    for layer_idx in range(self.num_cnn_layers):
      output = self.cnn[layer_idx](output)
      output = self.pooling[layer_idx](output)
    output_state = ([None for _ in range(self.num_layers)], [None for _ in range(self.num_layers)])
    for layer_idx in range(self.num_layers):
      output, output_state[0][layer_idx], output_state[1][layer_idx] = self.lstm[layer_idx](output, initial_state=(state[0][layer_idx], state[1][layer_idx]))
    output = self.dense(output)
    return output, output_state

class ADLSTM(AppDetector):
  def _init_handler(self, *args, **kwargs):
    self.batch_sz   = kwargs['batch_size']
    self.enc_units  = kwargs['num_hidden']
    self.num_layers = kwargs['num_layers']
    self.num_labels = kwargs['num_labels']

    #self.masking = tf.keras.layers.Masking(mask_value=np.nan)
    self.lstm = [tf.keras.layers.LSTM(
                                      self.enc_units,
                                      activation='tanh',
                                      #use_bias=,
                                      kernel_initializer='glorot_uniform',
                                      recurrent_initializer='glorot_uniform',
                                      #kernel_initializer='he_uniform',
                                      #recurrent_initializer='he_uniform',
                                      bias_initializer='zeros',
                                      #unit_forget_bias=,
                                      #kernel_regularizer=,
                                      #recurrent_regularizer=,
                                      #bias_regularizer=,
                                      #activity_regularizer=,
                                      #kernel_constraint=,
                                      #bias_constraint=,
                                      dropout=kwargs['p_dropout'],
                                      #recurrent_dropout=kwargs['p_dropout'],
                                      #implementation=2,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_dropout=0,
                                      go_backwards=False,
                                      stateful=True,
                                      unroll=False
                                     ) for _ in range(self.num_layers)]
    self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
    #self.dense = tf.keras.layers.Dense(
                                       self.num_labels,
                                       #activation='relu',
                                       activation='linear',
                                       #use_bias=,
                                       #kernel_initializer='he_uniform',
                                       kernel_initializer='glorot_uniform',
                                       bias_initializer='zeros',
                                       #kernel_regulariz
                                       #bias_regularizer=,
                                       #activity_regularizer=,
                                       #kernel_constraint=,
                                       #bias_constraint=,
                                       ),
                                       input_shape=(None,kwargs['max_timesteps'],self.enc_units)
                                      )

  def call(self, x, state):
    #output = self.masking(x)
    output = x
    output_state = ([None for _ in range(self.num_layers)], [None for _ in range(self.num_layers)])
    for layer_idx in range(self.num_layers):
      output, output_state[0][layer_idx], output_state[1][layer_idx] = self.lstm[layer_idx](output, initial_state=(state[0][layer_idx], state[1][layer_idx]))
    output = self.dense(output)
    return output, output_state

