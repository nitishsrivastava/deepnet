from layer import *

class SoftmaxLayer(Layer):
  def __init__(self, *args, **kwargs):
    super(SoftmaxLayer, self).__init__(*args, **kwargs)

  @classmethod
  def IsLayerType(cls, proto):
    return proto.hyperparams.activation == deepnet_pb2.Hyperparams.SOFTMAX

  def ApplyActivation(self):
    state = self.state
    temp = self.batchsize_temp

    state.max(axis=0, target=temp)
    state.add_row_mult(temp, -1)
    cm.exp(state)
    state.sum(axis=0, target=temp)
    state.div_by_row(temp)

  def Sample(self):
    self.state.perturb_prob_for_softmax_sampling(target=self.sample)
    self.sample.choose_max(axis=0)

  def ComputeDeriv(self):
    """Compute derivative w.r.t input given derivative w.r.t output."""
    raise Exception('Back prop through softmax not implemented.')

  def AllocateMemory(self, batchsize):
    super(SoftmaxLayer, self).AllocateMemory(batchsize)
    self.expansion_matrix = cm.CUDAMatrix(np.eye(self.numlabels))

  def AllocateBatchsizeDependentMemory(self, batchsize):
    super(SoftmaxLayer, self).AllocateBatchsizeDependentMemory(batchsize)
    dimensions = self.dimensions
    numlabels = self.numlabels
    self.data = cm.CUDAMatrix(np.zeros((dimensions, batchsize)))
    self.deriv = cm.CUDAMatrix(np.zeros((numlabels*dimensions, batchsize)))
    self.batchsize_temp = cm.CUDAMatrix(np.zeros((dimensions, batchsize)))
    if self.loss_function == deepnet_pb2.Layer.CROSS_ENTROPY:
      self.temp2 = cm.CUDAMatrix(np.zeros((dimensions, batchsize)))
      self.indices = cm.CUDAMatrix(np.zeros((1, dimensions * batchsize)))
      self.rowshift = cm.CUDAMatrix(
        numlabels*np.arange(dimensions * batchsize).reshape(1, -1))
    elif self.loss_function == deepnet_pb2.Layer.SQUARED_LOSS:
      self.expanded_batch = cm.CUDAMatrix(np.zeros((numlabels * dimensions, batchsize)))

  def GetData(self):
    self.expansion_matrix.select_columns(self.data, target=self.state)

  def GetLoss(self, get_deriv=False, **kwargs):
    """Compute loss and also deriv w.r.t to it if asked for.

    Compute the loss function. Targets should be in self.data, predictions
    should be in self.state.
    Args:
      get_deriv: If True, compute the derivative w.r.t the loss function and put
        it in self.deriv.
    """
    perf = deepnet_pb2.Metrics()
    perf.MergeFrom(self.proto.performance_stats)
    perf.count = self.batchsize
    tiny = self.tiny
    if self.loss_function == deepnet_pb2.Layer.CROSS_ENTROPY:
      temp2 = self.temp2
      temp = self.batchsize_temp
      batchsize = self.batchsize
      dimensions = self.dimensions
      numlabels = self.numlabels
      state = self.state
      data = self.data
      unitcell = self.unitcell
      indices = self.indices

      # Optimized for space to handle large number of labels in a softmax.
      data.reshape((1, batchsize * dimensions))
      data.add(self.rowshift, target=indices)
      state.reshape((numlabels, dimensions * batchsize))
      state.max(axis=0, target=temp2)
      state.reshape((1, batchsize * numlabels * dimensions))
      state.select_columns(indices, temp)
      temp2.subtract(temp)
      temp2.sign(target=temp2)
      temp2.sum(axis=1, target=unitcell)
      correct_preds = batchsize - unitcell.euclid_norm()
      if get_deriv:
        temp.subtract(1, target=temp2)
        state.set_selected_columns(indices, temp2)
        state.reshape((numlabels * dimensions, batchsize))
        self.deriv.assign(self.state)
      state.reshape((numlabels * dimensions, batchsize))
      temp.add(tiny)
      cm.log(temp)
      temp.sum(axis=1, target=unitcell)
      cross_entropy = unitcell.euclid_norm()
      perf.cross_entropy = cross_entropy
      perf.correct_preds = correct_preds
    elif self.loss_function == deepnet_pb2.Layer.SQUARED_LOSS:
      self.expansion_matrix.select_columns(self.data, target=self.expanded_batch)
      self.state.subtract(self.expanded_batch, target=self.deriv)
      error = self.deriv.euclid_norm()**2
      perf.error = error
    else:
      raise Exception('Unknown loss function for Softmax units.')
    return perf

  def GetSparsityDivisor(self):
    raise Exception('Sparsity not implemented for replicated softmax units.')

