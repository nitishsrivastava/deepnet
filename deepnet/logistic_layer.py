from layer import *

class LogisticLayer(Layer):
  def __init__(self, *args, **kwargs):
    super(LogisticLayer, self).__init__(*args, **kwargs)

  @classmethod
  def IsLayerType(cls, proto):
    return proto.hyperparams.activation == deepnet_pb2.Hyperparams.LOGISTIC

  def ApplyActivation(self):
    cm.sigmoid(self.state)

  def Sample(self):
    self.state.sample_bernoulli(target=self.sample)

  def ComputeDeriv(self):
    """Compute derivative w.r.t input given derivative w.r.t output."""
    self.deriv.apply_logistic_deriv(self.state)

  def GetLoss(self, get_deriv=False, acc_deriv=False, **kwargs):
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
      data = self.data
      state = self.state
      temp1 = self.statesize

      cm.cross_entropy_bernoulli(data, state, target=temp1, tiny=self.tiny)
      perf.cross_entropy = temp1.sum()
   
      cm.correct_preds(data, state, target=temp1, cutoff=0.5)
      perf.correct_preds = temp1.sum()

      if get_deriv:
        self.state.subtract(self.data, target=self.deriv)

    elif self.loss_function == deepnet_pb2.Layer.SQUARED_LOSS:
      target = self.statesize
      self.state.subtract(self.data, target=target)
      error = target.euclid_norm()**2
      perf.error = error
      if acc_deriv:
        self.deriv.add_mult(target, alpha=self.loss_weight)
      else:
        self.deriv.assign(target)
      if get_deriv:
        self.ComputeDeriv()
    else:
      raise Exception('Unknown loss function for logistic units.')

    return perf

  def GetSparsityDivisor(self):
    self.means_temp2.assign(1)
    self.means_temp2.subtract(self.means)
    self.means_temp2.mult(self.means)
    return self.means_temp2
  
