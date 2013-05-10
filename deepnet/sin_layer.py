from layer import *

class SinLayer(Layer):
  def __init__(self, *args, **kwargs):
    super(SinLayer, self).__init__(*args, **kwargs)

  @classmethod
  def IsLayerType(cls, proto):
    return proto.hyperparams.activation == deepnet_pb2.Hyperparams.SIN

  def ApplyActivation(self):
    self.backup_state.assign(self.state)
    cm.sin(self.state)

  def ComputeDeriv(self):
    """Compute derivative w.r.t input given derivative w.r.t output."""
    self.deriv.apply_sin_deriv(self.backup_state)

  def AllocateBatchsizeDependentMemory(self, batchsize):
    super(SinLayer, self).AllocateBatchsizeDependentMemory(batchsize)
    self.backup_state = cm.CUDAMatrix(np.zeros(self.statesize.shape))
