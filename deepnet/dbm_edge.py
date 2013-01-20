from edge import *

class DBMEdge(Edge):
  def LoadParams(self):
    super(DBMEdge, self).LoadParams()
    self.suff_stats = cm.empty((self.node1.numlabels * self.node1.dimensions,
                                self.node2.numlabels * self.node2.dimensions))

  def UpdateParams(self, step):
    """ Update the parameters associated with this edge.

    Update the weights and associated parameters using diff of pos and neg suff
    stats.
    Args:
      step: Training step.
    """
    logging.debug('UpdateParams in edge %s', self.name)
    h = self.hyperparams
    batchsize = self.node1.batchsize

    if h.momentum_change_steps > step:
      f = float(step) / h.momentum_change_steps
      momentum = (1.0 - f) * h.initial_momentum + f * h.final_momentum
    else:
      momentum = h.final_momentum
    if h.epsilon_decay == deepnet_pb2.Hyperparams.NONE:
      epsilon = h.base_epsilon
    elif h.epsilon_decay == deepnet_pb2.Hyperparams.INVERSE_T:
      epsilon = h.base_epsilon / (1 + float(step) / h.epsilon_decay_half_life)
    elif h.epsilon_decay == deepnet_pb2.Hyperparams.EXPONENTIAL:
      epsilon = h.base_epsilon / np.power(2, float(step) / h.epsilon_decay_half_life)
    if step < h.start_learning_after:
      epsilon = 0.0

    #w_delta = self.params['grad_weight']
    w_delta = self.grad_weight
    w = self.params['weight']
    w_delta.mult(momentum)
    if h.apply_l2_decay:
      w_delta.add_mult(w, -h.l2_decay)
    w_delta.add_mult(self.suff_stats, 1.0 / batchsize)
    w.add_mult(w_delta, epsilon)
    if h.apply_weight_norm:
      w.norm_limit(h.weight_norm, axis=0)

  def CollectSufficientStatistics(self, pos):
    logging.debug('Collecting suff stats %s', self.name)
    if self.node1.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.node1.state.div_by_row(self.node1.NN)
    if self.node2.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.node2.state.div_by_row(self.node2.NN)
    if pos:
      h1 = self.node1.hyperparams
      h2 = self.node2.hyperparams
      if h1.sparsity:
        self.node1.state.add_col_mult(self.node1.means_temp, -h1.sparsity_cost)
      if h2.sparsity:
        self.node2.state.add_col_mult(self.node2.means_temp, -h2.sparsity_cost)
      cm.dot(self.node1.state, self.node2.state.T, target=self.suff_stats)
      if h1.sparsity:
        self.node1.state.add_col_mult(self.node1.means_temp, h1.sparsity_cost)
      if h2.sparsity:
        self.node2.state.add_col_mult(self.node2.means_temp, h2.sparsity_cost)
    else:
      self.suff_stats.add_dot(self.node1.state, self.node2.state.T, mult=-1.0)
    if self.node1.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.node1.state.mult_by_row(self.node1.NN)
    if self.node2.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.node2.state.mult_by_row(self.node2.NN)
