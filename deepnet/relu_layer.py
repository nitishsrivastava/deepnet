from layer import *

class ReluLayer(Layer):
  def __init__(self, *args, **kwargs):
    super(ReluLayer, self).__init__(*args, **kwargs)

  @classmethod
  def IsLayerType(cls, proto):
    return proto.hyperparams.activation == deepnet_pb2.Hyperparams.RECTIFIED_LINEAR

  def ApplyActivation(self, neg=False):
    if neg:
      state = self.neg_state
    else:
      state = self.state
    state.lower_bound(0)

  def Sample(self, neg=False):
    if neg:
      sample = self.neg_sample
      state = self.neg_state
    else:
      sample = self.sample
      state = self.state
    state.sample_gaussian(target=sample, mult=1.0)
    sample.lower_bound(0)

  def ComputeDeriv(self):
    """Compute derivative w.r.t input given derivative w.r.t output."""
    self.deriv.apply_rectified_linear_deriv(self.state)

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
    if self.loss_function == deepnet_pb2.Layer.SQUARED_LOSS:
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
      raise Exception('Unknown loss function for ReLU units.')
    return perf

