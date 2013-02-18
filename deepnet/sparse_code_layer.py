from layer import *

class SparseCodeLayer(Layer):

  def AllocateBatchsizeDependentMemory(self, batchsize):
    super(SparseCodeLayer, self).AllocateBatchsizeDependentMemory(batchsize)
    self.approximator = cm.empty(self.state.shape)
    self.temp3 = cm.empty(self.state.shape)
    self.grad = cm.empty(self.state.shape)
    self.grad_scale = cm.CUDAMatrix(np.zeros((self.state.shape[0], 1)))
    self.m_by_m = cm.empty((self.state.shape[0], self.state.shape[0]))

  def ApplyActivation(self, state):
    if self.activation == deepnet_pb2.Hyperparams.LOGISTIC:
      cm.sigmoid(state)
    elif self.activation == deepnet_pb2.Hyperparams.TANH:
      cm.tanh(state)
    elif self.activation == deepnet_pb2.Hyperparams.RECTIFIED_LINEAR:
      state.greater_than(0, target=self.temp)
      state.mult(self.temp)
    elif self.activation == deepnet_pb2.Hyperparams.RECTIFIED_LINEAR_SMOOTH:
      cm.log_1_plus_exp(state)
    elif self.activation == deepnet_pb2.Hyperparams.LINEAR:
      pass

  def ComputeDeriv(self, state):
    """Compute derivative w.r.t input given derivative w.r.t output."""
    if self.activation == deepnet_pb2.Hyperparams.LOGISTIC:
      self.deriv.apply_logistic_deriv(state)
    elif self.activation == deepnet_pb2.Hyperparams.TANH:
      self.deriv.apply_tanh_deriv(state)
      if self.hyperparams.dropout:
        self.deriv.mult(self.mask)
    elif self.activation == deepnet_pb2.Hyperparams.RECTIFIED_LINEAR:
      self.deriv.apply_rectified_linear_deriv(state)
    elif self.activation == deepnet_pb2.Hyperparams.RECTIFIED_LINEAR_SMOOTH:
      self.deriv.apply_rectified_linear_smooth_deriv(state)
    elif self.activation == deepnet_pb2.Hyperparams.LINEAR:
      if self.hyperparams.dropout:
        self.deriv.mult(self.mask)
    elif self.activation == deepnet_pb2.Hyperparams.SOFTMAX:
      raise Exception('Not implemented.')
    else:
      raise Exception('Unknown activation.')



