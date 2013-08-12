from choose_matrix_library import *
import deepnet_pb2
import logging
import numpy as np
import os.path
import util
import visualize
import pdb

class Parameter(object):

  def __init__(self):
    self.num_shares = 1
    self.num_grads_received = 0
    self.transpose = False

  def SaveParameters(self):
    for param in self.proto.param:
      param.mat = util.NumpyAsParameter(self.params[param.name].asarray())

  def LoadParams(self, proto, t_op=None, tied_to=None):
    """Load the parameters for this edge.

    Load the parameters if present in self.proto. Otherwise initialize them
    appropriately.
    """
    param_names = [param.name for param in proto.param]
    for param in proto.param:
      assert param.dimensions, 'Empty dimensions'
      if tied_to:
        if self.transpose:
          self.params[param.name] = tied_to.params[param.name].T
        else:
          self.params[param.name] = tied_to.params[param.name]
        mat = self.params[param.name]
      else:
        if param.mat:
          mat = util.ParameterAsNumpy(param)
        else:
          mat = self.InitializeParameter(param)
        self.params[param.name] = cm.CUDAMatrix(mat)

  def InitializeParameter(self, param):
    if param.initialization == deepnet_pb2.Parameter.CONSTANT:
      return np.zeros(tuple(param.dimensions)) + param.constant
    elif param.initialization == deepnet_pb2.Parameter.DENSE_GAUSSIAN:
      return param.sigma * np.random.randn(*tuple(param.dimensions))
    elif param.initialization == deepnet_pb2.Parameter.DENSE_UNIFORM:
      return param.sigma * (2 * np.random.rand(*tuple(param.dimensions)) - 1)
    elif param.initialization == deepnet_pb2.Parameter.DENSE_GAUSSIAN_SQRT_FAN_IN:
      assert len(param.dimensions) > 1
      if param.conv or param.local:
        fan_in = np.prod(param.dimensions[0])
      else:
        fan_in = np.prod(param.dimensions[1])
      stddev = param.sigma / np.sqrt(fan_in)
      return stddev * np.random.randn(*tuple(param.dimensions))
    elif param.initialization == deepnet_pb2.Parameter.DENSE_UNIFORM_SQRT_FAN_IN:
      assert len(param.dimensions) > 1
      if param.conv or param.local:
        fan_in = np.prod(param.dimensions[0])
      else:
        fan_in = np.prod(param.dimensions[1])
      stddev = param.sigma / np.sqrt(fan_in)
      return stddev * (2 * np.random.rand(*tuple(param.dimensions)) - 1)
    elif param.initialization == deepnet_pb2.Parameter.PRETRAINED:
      return self.LoadPretrained(param)
    else:
      raise Exception('Unknown parameter initialization.')

  def LoadPretrained(self, param):
    pass

  def GetGlobalInfo(self, net):
    pass

  def ApplyL2Decay(self, w_delta, w, lambdaa, **kwargs):
    w_delta.add_mult(w, lambdaa)

  def Update(self, param_name, step, no_reg=False):
    h = self.hyperparams
    momentum, epsilon = self.GetMomentumAndEpsilon(step)

    w = self.params[param_name]  # Parameter to be updated.
    w_delta = self.gradient_history  # Previous update.
    gradient = self.gradient  # Current gradient.

    # Compute update.
    if h.adapt == deepnet_pb2.Hyperparams.NONE:
      w_delta.mult(momentum)
      if not no_reg and h.apply_l2_decay:
        self.ApplyL2Decay(w_delta, w, h.l2_decay, step=step, eps=epsilon, mom=momentum)
      if not no_reg and h.apply_l1_decay and step > h.apply_l1decay_after:
        w_delta.add_mult_sign(w, h.l1_decay)
    else:
      raise Exception('Not implemented.')
    w_delta.add_mult(gradient)

    # Apply update.
    w.add_mult(w_delta, -epsilon)
    if not no_reg and h.apply_weight_norm:
      w.norm_limit(h.weight_norm, axis=0)

    # Reset.
    self.num_grads_received = 0
    gradient.assign(0)

  def GetMomentumAndEpsilon(self, step):
    """
    if h.momentum_change_steps > step:
      f = float(step) / h.momentum_change_steps
      momentum = (1.0 - f) * h.initial_momentum + f * h.final_momentum
    else:
      momentum = h.final_momentum
    """
    h = self.hyperparams
    momentum = h.final_momentum - (h.final_momentum - h.initial_momentum)*np.exp(-float(step)/h.momentum_change_steps)
    epsilon = h.base_epsilon
    if h.epsilon_decay == deepnet_pb2.Hyperparams.INVERSE_T:
      epsilon = h.base_epsilon / (1 + float(step) / h.epsilon_decay_half_life)
    elif h.epsilon_decay == deepnet_pb2.Hyperparams.EXPONENTIAL:
      epsilon = h.base_epsilon / np.power(2, float(step) / h.epsilon_decay_half_life)
    if step < h.start_learning_after:
      epsilon = 0.0
    return momentum, epsilon

