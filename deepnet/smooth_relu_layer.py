from relu_layer import *

class SmoothReluLayer(ReluLayer):
  def __init__(self, *args, **kwargs):
    super(SmoothReluLayer, self).__init__(*args, **kwargs)

  @classmethod
  def IsLayerType(cls, proto):
    return proto.hyperparams.activation == deepnet_pb2.Hyperparams.RECTIFIED_LINEAR_SMOOTH

  def ApplyActivation(self):
    if neg:
      state = self.neg_state
    else:
      state = self.state
    cm.log_1_plus_exp(state)

  def ComputeDeriv(self):
    """Compute derivative w.r.t input given derivative w.r.t output."""
    self.deriv.apply_rectified_linear_smooth_deriv(self.state)
