from layer import *

class ReplicatedSoftmaxLayer(Layer):
  def __init__(self, *args, **kwargs):
    super(ReplicatedSoftmaxLayer, self).__init__(*args, **kwargs)

  @classmethod
  def IsLayerType(cls, proto):
    return proto.hyperparams.activation == \
        deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX

  def ApplyActivation(self):
    state = self.state
    temp = self.batchsize_temp

    state.max(axis=0, target=temp)
    state.add_row_mult(temp, -1)
    cm.exp(state)
    state.sum(axis=0, target=temp)
    self.NN.divide(temp, target=temp)
    state.mult_by_row(temp)

  def Sample(self):
    sample = self.sample
    state = self.state
    use_lightspeed = False
    if use_lightspeed:  # Do sampling on cpu.
      temp = self.expanded_batch
      state.sum(axis=0, target=self.temp)
      state.div_by_row(self.temp, target=temp)
      probs_cpu = temp.asarray().astype(np.float64)
      numsamples = self.NN.asarray()
      samples_cpu = lightspeed.SampleSoftmax(probs_cpu, numsamples)
      sample.overwrite(samples_cpu.astype(np.float32))
    else:
      if self.proto.hyperparams.adaptive_prior > 0:
        sample.assign(0)
        temp_sample = self.expanded_batch
        numsamples = int(self.proto.hyperparams.adaptive_prior)
        for i in range(numsamples):
          state.perturb_prob_for_softmax_sampling(target=temp_sample)
          temp_sample.choose_max_and_accumulate(sample)
      else:
        NN = self.NN.asarray().reshape(-1)
        numdims, batchsize = self.state.shape
        max_samples = self.big_sample_matrix.shape[1]
        for i in range(batchsize):
          nn = NN[i]
          factor = 1
          if nn > max_samples:
            nn = max_samples
            factor = float(nn) / max_samples
          samples = self.big_sample_matrix.slice(0, nn)
          samples.assign(0)
          samples.add_col_vec(self.state.slice(i, i+1))
          samples.perturb_prob_for_softmax_sampling()
          samples.choose_max(axis=0)
          samples.sum(axis=1, target=sample.slice(i, i+1))
          if factor > 1:
            sample.slice(i, i+1).mult(factor)

  def ComputeDeriv(self):
    """Compute derivative w.r.t input given derivative w.r.t output."""
    raise Exception('Back prop through replicated softmax not implemented.')

  def AllocateMemory(self, batchsize):
    super(ReplicatedSoftmaxLayer, self).AllocateMemory(batchsize)
    self.expansion_matrix = cm.CUDAMatrix(np.eye(self.numlabels))
    self.big_sample_matrix = cm.empty((self.numlabels * self.dimensions, 1000))

  def AllocateBatchsizeDependentMemory(self, batchsize):
    super(ReplicatedSoftmaxLayer, self).AllocateBatchsizeDependentMemory(batchsize)
    dimensions = self.dimensions
    numlabels = self.numlabels
    self.expanded_batch = cm.CUDAMatrix(np.zeros((numlabels * dimensions, batchsize)))
    self.batchsize_temp = cm.CUDAMatrix(np.zeros((dimensions, batchsize)))
    if self.is_input or self.is_initialized or self.is_output:
      self.data = cm.CUDAMatrix(np.zeros((numlabels * dimensions, batchsize)))
    self.NN = cm.CUDAMatrix(np.ones((1, batchsize)))
    self.counter = cm.empty(self.NN.shape)
    self.count_filter = cm.empty(self.NN.shape)

  def ResetState(self, rand=False):
    if self.hyperparams.normalize:
      self.NN.assign(self.hyperparams.normalize_to)
    else:
      self.NN.assign(1)
    super(ReplicatedSoftmaxLayer, self).ResetState(rand=rand)

  def GetData(self):
    self.state.assign(self.data)
    h = self.hyperparams
    self.state.sum(axis=0, target=self.NN)
    self.NN.add(self.tiny)  # To deal with documents of 0 words.
    if h.multiplicative_prior > 0:
      self.NN.mult(1 + h.multiplicative_prior)
      self.state.mult(1 + h.multiplicative_prior)
    if h.additive_prior > 0:
      self.state.div_by_row(self.NN)
      self.NN.add(h.additive_prior)
      self.state.mult_by_row(self.NN)
    if h.adaptive_prior > 0:
      self.state.div_by_row(self.NN)
      self.state.mult(h.adaptive_prior)
      self.NN.assign(h.adaptive_prior)

  def GetLoss(self, get_deriv=False):
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
    temp = self.batchsize_temp
    if self.loss_function == deepnet_pb2.Layer.SQUARED_LOSS:
      if get_deriv:
        target = self.deriv
      else:
        target = self.statesize
      if self.hyperparams.normalize_error:
        self.data.sum(axis=0, target=temp)
        temp.add(self.tiny)
        self.data.div_by_row(temp, target=target)
        self.state.div_by_row(self.NN, target=self.expanded_batch)
        target.subtract(self.expanded_batch)
      else:
        self.data.sum(axis=0, target=temp)
        temp.add(self.tiny)
        self.state.div_by_row(temp, target=target)
        target.subtract(self.data)
      error = target.euclid_norm()**2
      perf.error = error
    else:
      raise Exception('Unknown loss function for Replicated Softmax units.')
    return perf

  def GetSparsityDivisor(self):
    raise Exception('Sparsity not implemented for replicated softmax units.')

  def CollectSufficientStatistics(self, neg=False):
    """Collect sufficient statistics for this layer."""
    h = self.hyperparams
    self.state.div_by_row(self.NN)
    if not neg:
      self.state.sum(axis=1, target=self.suff_stats)
    else:
      self.suff_stats.add_sums(self.state, axis=1, mult=-1.0)
    self.state.mult_by_row(self.NN)
