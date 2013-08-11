from edge import *

class TransferEdge(Edge):
  @classmethod
  def IsEdgeType(cls, proto):
    return proto.hyperparams.shared_prior

  def LoadParams(self, proto, **kwargs):
    super(TransferEdge, self).LoadParams(proto, **kwargs)
    fname = os.path.join(self.prefix, self.hyperparams.shared_prior_file)
    self.shared_prior_cost = self.hyperparams.shared_prior_cost
    sc = np.load(fname).reshape(1, -1)
    self.superclasses = cm.CUDAMatrix(sc)
    self.diff = cm.empty(self.params['weight'].shape)
    self.expanded_prior_mean = cm.empty(self.params['weight'].shape)
    self.prior_mean = cm.empty((self.diff.shape[0], sc.max() +1)) 

  def GetGlobalInfo(self, net):
    pass
    #edge_name = self.hyperparams.shared_prior_edge
    #prior_edge = next(e for e in net.edge if e.name == edge_name)
    #self.prior_mean = prior_edge.params['weight']

  def ApplyL2Decay(self, w_delta, w, lambdaa, step=0, **kwargs):
    if step % 50 == 0:
      w.accumulate_columns(self.superclasses, self.prior_mean, avg=True)
      self.prior_mean.expand(self.superclasses, target=self.expanded_prior_mean)
    diff = self.diff
    w.subtract(self.expanded_prior_mean, target=diff)
    w_delta.add_mult(diff, lambdaa)
    w_delta.add_mult(self.expanded_prior_mean, self.shared_prior_cost)

"""
class TransferEdge(Edge):
  @classmethod
  def IsEdgeType(cls, proto):
    return proto.hyperparams.shared_prior

  def LoadParams(self, proto, **kwargs):
    super(TransferEdge, self).LoadParams(proto, **kwargs)
    fname = os.path.join(self.prefix, self.hyperparams.shared_prior_file)
    self.superclasses = cm.CUDAMatrix(np.load(fname).reshape(1, -1))
    self.diff = cm.empty(self.params['weight'].shape)

  def GetGlobalInfo(self, net):
    edge_name = self.hyperparams.shared_prior_edge
    prior_edge = next(e for e in net.edge if e.name == edge_name)
    self.prior_mean = prior_edge.params['weight']

  def ApplyL2Decay(self, w_delta, w, lambdaa):
    diff = self.diff
    w.expand_and_add(self.prior_mean, self.superclasses, target=diff, mult=-1.0)
    w_delta.add_mult(diff, lambdaa)
"""
