from layer import *

class LinearLayer(Layer):
  def __init__(self, *args, **kwargs):
    super(LinearLayer, self).__init__(*args, **kwargs)

  @classmethod
  def IsLayerType(cls, proto):
    return proto.hyperparams.activation == deepnet_pb2.Hyperparams.LINEAR

  def ApplyActivation(self):
    pass

  def Sample(self):
    sample = self.sample
    state = self.state
    #sample.assign(state)
    #state.sample_gaussian(target=sample, mult=0.01)
    if self.learn_precision:
      sample.fill_with_randn()
      sample.div_by_col(self.params['precision'])
      sample.add(state)

  def ComputeDeriv(self):
    """Compute derivative w.r.t input given derivative w.r.t output."""
    if self.hyperparams.dropout:
      self.deriv.mult(self.mask)

  def GetData(self):
    self.state.assign(self.data)
    if 'precision' in self.params:
      self.state.mult_by_col(self.params['precision'])

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
    if self.loss_function == deepnet_pb2.Layer.SQUARED_LOSS:
      if get_deriv:
        target = self.deriv
      else:
        target = self.statesize
      if 'precision' in self.params:
        self.data.mult_by_col(self.params['precision'], target=target)
        target.subtract(self.state)
      else:
        self.state.subtract(self.data, target=target)
      error = target.euclid_norm()**2
      perf.error = error
      if get_deriv:
        self.ComputeDeriv()
    elif self.loss_function == deepnet_pb2.Layer.HINGE_LOSS:
      pass
    else:
      raise Exception('Unknown loss function for linear units.')
    return perf

  def GetSparsityDivisor(self):
    self.means_temp2.assign(1)

