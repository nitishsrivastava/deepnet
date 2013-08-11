from edge import *

class SoftTransferEdge(Edge):
  @classmethod
  def IsEdgeType(cls, proto):
    return proto.hyperparams.soft_shared_prior


  def InitializeSoftWeights(self, superclasses):
    numsuperclasses = superclasses.max() + 1
    numclasses = len(superclasses)
    sw = np.zeros((numclasses, numclasses))
    for k in range(numsuperclasses):
      indices = (superclasses == k).nonzero()[0]
      for i in indices:
        sw[i, indices] = 1
        sw[i, i] = 0
    sw /= sw.sum(axis=1) + 1e-10
    return sw

  def LoadParams(self, proto, **kwargs):
    super(SoftTransferEdge, self).LoadParams(proto, **kwargs)
    self.shared_prior_cost = self.hyperparams.shared_prior_cost

    if self.hyperparams.shared_prior_file:
      fname = os.path.join(self.prefix, self.hyperparams.shared_prior_file)
      sc = np.load(fname).reshape(-1)
      sw = self.InitializeSoftWeights(sc)
      if 'soft_weight' in self.params:
        self.params['soft_weight'].overwrite(sw)
      else:
        self.params['soft_weight'] = cm.CUDAMatrix(sw)
    if self.hyperparams.label_freq_file:
      fname = os.path.join(self.prefix, self.hyperparams.label_freq_file)
      label_freq = np.load(fname).reshape(1, -1)
      self.label_freq = cm.CUDAMatrix(label_freq)
    else:
      self.label_freq = None

    w = self.params['weight']
    self.prior_mean = cm.empty(w.shape)
    self.diff = cm.empty(w.shape)

  def AllocateMemory(self):
    super(SoftTransferEdge, self).AllocateMemory()
    self.transform_gradient_history = cm.CUDAMatrix(np.zeros(self.params['soft_weight'].shape))

  def ApplyL2Decay(self, w_delta, w, lambdaa, step=0, eps=0.0, mom=0.0, **kwargs):
    diff = self.diff
    transform = self.params['soft_weight']
    transform_delta = self.transform_gradient_history
    pm = self.prior_mean

    cm.dot(w, transform, target=pm)
    w.subtract(pm, target=diff)
    if self.label_freq is not None:
      diff.div_by_row(self.label_freq)
    w_delta.add_mult(diff, lambdaa)

    transform_delta.mult(mom)
    transform_delta.add_dot(w.T, diff, mult=-lambdaa)  # gradient for the transform.
    transform_delta.add_mult_sign(transform, self.shared_prior_cost) # L1-Decay the transform.
    transform_delta.mult_diagonal(0)  # set diagonal to zero.
    transform.add_mult(transform_delta, -eps)

  def Show(self):
    if not self.hyperparams.enable_display:
      return
    visualize.show(self.params['soft_weight'].asarray(), self.fig, title='Sharing weights')

