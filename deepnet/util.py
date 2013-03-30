"""Utility functions for loading/saving models."""

import cPickle as pickle
import deepnet_pb2
import gzip
import numpy as np
import os.path
import shutil
import time
import pdb
from google.protobuf import text_format

def ParameterAsNumpy(param):
  """Converts a serialized parameter string into a numpy array."""
  return param.mult_factor * np.fromstring(param.mat, dtype='float32').reshape(
    *tuple(param.dimensions))

def NumpyAsParameter(numpy_array):
  """Converts a numpy array into a serialized parameter string."""
  assert numpy_array.dtype == 'float32', 'Saved arrays should be float32.'
  return numpy_array.tostring()

def WriteCheckpointFile(net, t_op, best=False):
  """Writes out the model to disk."""
  ckpt_dir = os.path.join(t_op.checkpoint_prefix, t_op.checkpoint_directory)
  if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
  if best:
    tag = 'BEST'
    checkpoint_file = '%s_%s' % (net.name, tag)
    checkpoint_file = os.path.join(ckpt_dir, checkpoint_file)
    print 'Writing current best model %s' % checkpoint_file
    f = gzip.open(checkpoint_file, 'wb')
    f.write(net.SerializeToString())
    f.close()
  else:
    tag = 'LAST'
    checkpoint_file = '%s_%s' % (net.name, time.strftime('%s'))
    checkpoint_file = os.path.join(ckpt_dir, checkpoint_file)
    print 'Writing checkpoint %s' % checkpoint_file
    f = gzip.open(checkpoint_file, 'wb')
    f.write(net.SerializeToString())
    f.close()
    checkpoint_file_LAST = '%s_%s' % (net.name, tag)
    checkpoint_file_LAST = os.path.join(ckpt_dir, checkpoint_file_LAST)
    shutil.copyfile(checkpoint_file, checkpoint_file_LAST)

  # Save the t_op.
  checkpoint_file_op = '%s_train_op_%s' % (net.name, tag)
  checkpoint_file = os.path.join(ckpt_dir, checkpoint_file_op)
  f = gzip.open(checkpoint_file, 'wb')
  f.write(t_op.SerializeToString())
  f.close()

def ReadOperation(proto_file):
  protoname, ext = os.path.splitext(proto_file)
  proto = deepnet_pb2.Operation()
  if ext == '.pbtxt':
    proto_pbtxt = open(proto_file, 'r')
    text_format.Merge(proto_pbtxt.read(), proto)
  else:
    f = gzip.open(proto_file, 'rb')
    proto.ParseFromString(f.read())
    f.close()
  return proto

def ReadModel(proto_file):
  protoname, ext = os.path.splitext(proto_file)
  proto = deepnet_pb2.Model()
  if ext == '.pbtxt':
    proto_pbtxt = open(proto_file, 'r')
    text_format.Merge(proto_pbtxt.read(), proto)
  else:
    f = gzip.open(proto_file, 'rb')
    proto.ParseFromString(f.read())
    f.close()
  return proto

def WritePbtxt(output_file, pb):
  with open(output_file, 'w') as f:
    text_format.PrintMessage(pb, f)

def ReadData(proto_file):
  protoname, ext = os.path.splitext(proto_file)
  proto = deepnet_pb2.Dataset()
  if ext == '.pbtxt':
    proto_pbtxt = open(proto_file, 'r')
    text_format.Merge(proto_pbtxt.read(), proto)
  else:
    f = open(proto_file, 'rb')
    proto.ParseFromString(f.read())
    f.close()
  return proto

def CopyData(data):
  copy = deepnet_pb2.Dataset.Data()
  copy.CopyFrom(data)
  return copy

def CopyDataset(data):
  copy = deepnet_pb2.Dataset()
  copy.CopyFrom(data)
  return copy

def CopyOperation(op):
  copy = deepnet_pb2.Operation()
  copy.CopyFrom(op)
  return copy

def CopyModel(model):
  copy = deepnet_pb2.Model()
  copy.CopyFrom(model)
  return copy

def CopyLayer(layer):
  copy = deepnet_pb2.Layer()
  copy.CopyFrom(layer)
  return copy

def GetPerformanceStats(stat, prefix=''):
  s = ''
  if stat.compute_cross_entropy:
    s += ' %s_CE: %.3f' % (prefix, stat.cross_entropy / stat.count)
  if stat.compute_correct_preds:
    s += ' %s_Acc: %.3f (%d/%d)' % (
      prefix, stat.correct_preds/stat.count, stat.correct_preds, stat.count)
  if stat.compute_error:
    s += ' %s_E: %.5f' % (prefix, stat.error / stat.count)
  if stat.compute_MAP and prefix != 'T':
    s += ' %s_MAP: %.3f' % (prefix, stat.MAP)
  if stat.compute_prec50 and prefix != 'T':
    s += ' %s_prec50: %.3f' % (prefix, stat.prec50)
  if stat.compute_sparsity:
    s += ' %s_sp: %.3f' % (prefix, stat.sparsity / stat.count)
  return s

def Accumulate(acc, perf):
 acc.count += perf.count
 acc.cross_entropy += perf.cross_entropy
 acc.error += perf.error
 acc.correct_preds += perf.correct_preds
 acc.sparsity += perf.sparsity

def CreateLayer(layer_class, proto, t_op):
  for cls in layer_class.__subclasses__():
    if cls.IsLayerType(proto):
      return cls(proto, t_op)
    l = CreateLayer(cls, proto, t_op)
    if l is not None:
      return l
  return None


def GetMomentumAndEpsilon(h, step):
  """
  if h.momentum_change_steps > step:
    f = float(step) / h.momentum_change_steps
    momentum = (1.0 - f) * h.initial_momentum + f * h.final_momentum
  else:
    momentum = h.final_momentum
  """
  momentum = h.final_momentum - (h.final_momentum - h.initial_momentum)*np.exp(-float(step)/h.momentum_change_steps)
  epsilon = h.base_epsilon
  if h.epsilon_decay == deepnet_pb2.Hyperparams.INVERSE_T:
    epsilon = h.base_epsilon / (1 + float(step) / h.epsilon_decay_half_life)
  elif h.epsilon_decay == deepnet_pb2.Hyperparams.EXPONENTIAL:
    epsilon = h.base_epsilon / np.power(2, float(step) / h.epsilon_decay_half_life)
  if step < h.start_learning_after:
    epsilon = 0.0
  return momentum, epsilon




# For Navdeep's data.
def save(fname, var_list, source_dict):
    var_list = [var.strip() for var in var_list.split() if len(var.strip())>0]
    fo = gzip.GzipFile(fname, 'wb')
    pickle.dump(var_list, fo)
    for var in var_list:
        pickle.dump(source_dict[var], fo, protocol=2)
    fo.close()

def load(fname, target_dict, verbose = False):
    fo = gzip.GzipFile(fname, 'rb')
    var_list = pickle.load(fo)
    if verbose:
        print var_list
    for var in var_list:
        target_dict[var] = pickle.load(fo)
    fo.close()
