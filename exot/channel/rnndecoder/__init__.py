"""Legacy modules used for the RNN decoder for the frequency covert channel"""
import importlib

from .dataaugmentation import DataAugmentation

if importlib.util.find_spec("tensorflow") is not None:
  import tensorflow as tf
  tfversion = tf.__version__.split('.')
  if int(tfversion[0]) == 1 and int(tfversion[1]) == 2:
    from .evaluate import RNNdecoderEval
    from .train    import RNNdecoderTrain
  else:
      print("Wrong tensorflow version for RNN decoder, need version 1.2.0, has {tf.__version__}!")

__all__ = (
    "RNNdecoderEval",
    "RNNdecoderTrain",
)

