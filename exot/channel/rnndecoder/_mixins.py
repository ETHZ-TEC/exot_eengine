"""TODO
@author: hex
         miedlp (2018-06-13)
"""
# --------------------------------------------------------------------------------------------------
# Imports
import os
import tensorflow as tf
import numpy      as np
import os
import pickle

from datetime import datetime

# --------------------------------------------------------------------------------------------------
# Cut minibatches to equal length
def cut_minibatch(label, freqv, minibatch_length):
  cut_length = len(freqv) % minibatch_length
  zip_array  = list(zip(label, freqv))
  zip_array.sort(key=lambda x: len(x[1]))
  if(cut_length == 0):
    return(label, freqv)
  else:
    del zip_array[-cut_length:]
    return zip(*zip_array)

# --------------------------------------------------------------------------------------------------
# Count how many values in a tensor a equal to a value val
def tf_count(t, val):
  cnt_equal_elem = tf.equal(t, val)
  cnt_equal_elem = tf.cast(cnt_equal_elem, tf.int32)
  return tf.reduce_sum(cnt_equal_elem, 1)

# --------------------------------------------------------------------------------------------------
# Make data zero mean and with unit standard deviation
def stdr(dataseq):
  flat_dataseq = np.array([y for x in dataseq for y in x])
  dataseq      = np.array(dataseq)
  mean         = np.mean(flat_dataseq)
  std          = np.std(flat_dataseq)
  return (dataseq - mean) / std, mean, std

# --------------------------------------------------------------------------------------------------
# Make data zero mean and with unit variance with given mean and standard deviation
def stdr_val(val_freqv, mean, std):
  return (val_freqv - mean) / std

# --------------------------------------------------------------------------------------------------
# Separate the training and the validation dataset
def separate_val(freqv, label):
  val_index = np.random.choice(len(freqv), len(freqv) // 10, replace=False)
  val_freqv = freqv[val_index]
  val_label = label[val_index]
  freqv=np.delete(freqv, val_index, 0)
  label=np.delete(label, val_index, 0)
  return freqv, val_freqv, label, val_label, val_index

