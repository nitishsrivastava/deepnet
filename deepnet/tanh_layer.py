from layer import *

class TanhLayer(Layer):
  def __init__(self, *args, **kwargs):
    super(TanhLayer, self).__init__(*args, **kwargs)

  @classmethod
  def IsLayerType(cls, proto):
    return proto.hyperparams.activation == deepnet_pb2.Hyperparams.TANH

  def ApplyActivation(self):
    cm.tanh(self.state)

  def Sample(self):
    self.state.sample_bernoulli_tanh(target=self.sample)

  def ComputeDeriv(self):
    """Compute derivative w.r.t input given derivative w.r.t output."""
    self.deriv.apply_tanh_deriv(self.state)
    if self.hyperparams.dropout:
      self.deriv.mult(self.mask)

  def GetLoss(self, get_deriv=False, **kwargs):
    """Computes loss.

    Computes the loss function. Assumes target is in self.data and predictions
    are in self.state.
    Args:
      get_deriv: If True, computes the derivative of the loss function w.r.t the
      inputs to this layer and puts the result in self.deriv.
    """
    perf = deepnet_pb2.Metrics()
    perf.MergeFrom(self.proto.performance_stats)
    perf.count = self.batchsize
    if self.loss_function == deepnet_pb2.Layer.SQUARED_LOSS:
      self.state.subtract(self.data, target=self.deriv)
      error = self.deriv.euclid_norm()**2
      perf.error = error
      if get_deriv:
        self.ComputeDeriv()
    else:
      raise Exception('Unknown loss function for tanh units.')
    return perf

  def GetSparsityDivisor(self):
    self.means_temp2.assign(1)
    self.means_temp2.subtract(self.means, target=self.means_temp)
    self.means_temp2.add(self.means)
    self.means_temp2.mult(self.means_temp)
    return self.means_temp2

