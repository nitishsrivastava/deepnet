"""Implements a layer of a Deep Boltzmann Machine."""
from layer import *
import pdb

class DBMLayer(Layer):

  def __init__(self, *args, **kwargs):
    self.phases_tied = False
    self.suff_stats = None
    self.learn_precision = False
    self.pos_phase = True
    self.sample_input = False
    super(DBMLayer, self).__init__(*args, **kwargs)

  def LoadParams(self, proto):
    super(DBMLayer, self).LoadParams(proto)
    self.suff_stats = cm.empty((proto.numlabels * proto.dimensions, 1))
    self.learn_precision = self.activation == deepnet_pb2.Hyperparams.LINEAR and self.hyperparams.learn_precision
    if self.learn_precision:
      self.suff_stats2 = cm.empty((proto.numlabels * proto.dimensions, 1))
    self.fig_neg = visualize.GetFigId()
    self.fig_precision = visualize.GetFigId()
    self.sample_input = self.hyperparams.sample_input
  
  def Show(self):
    """Displays useful statistics about the layer."""
    if not self.proto.hyperparams.enable_display:
      return
    if self.is_input:
      visualize.display_hidden(self.data.asarray(), self.fig, title=self.name)
      #visualize.display_w(self.neg_state.asarray(), self.proto.shape[0],
      #                    10, self.batchsize/10, self.fig, title='data')
      #visualize.display_w(self.params['bias'].asarray(),
      #                    self.proto.shape[0], 1, 1, self.fig,
      #                    title='bias')
      #visualize.display_w(self.params['precision'].asarray(),
      #                    self.proto.shape[0], 1, 1, self.fig_precision,
      #                    title='precision')
    else:
      visualize.display_hidden(self.pos_state.asarray(), self.fig_neg, title=self.name + "_positive")
      #visualize.display_hidden(self.neg_state.asarray(), 2*self.fig_neg, title=self.name + "_negative")
      """
      visualize.display_w(self.pos_state.asarray(), self.proto.shape[0],
                          self.batchsize, 1, self.fig,
                          title=self.name + "_positive", vmin=0, vmax=1)
      visualize.display_w(self.neg_sample.asarray(), self.proto.shape[0],
                          self.batchsize, 1, self.fig_neg,
                          title=self.name + "_negative", vmin=0, vmax=1)
      """
  def SetPhase(self, pos=True):
    """Setup required before starting a phase.

    This method makes 'state' and 'sample' point to the right variable depending
    on the phase.
    """
    logging.debug('SetPhase in %s', self.name)
    if pos:
      self.pos_phase = True
      self.state = self.pos_state
      self.sample = self.pos_sample
    else:
      self.pos_phase = False
      self.state = self.neg_state
      self.sample = self.neg_sample

  def TiePhases(self):
    """Ties the variables used in pos and neg phases.

    This is done to save memory when doing CD. Since the Markov chain is not run
    persistently, the neg state need not be preserved after each cycle.
    """
    self.phases_tied = True
    if self.neg_state != self.pos_state:
      self.neg_state.free_device_memory()
      self.neg_state = self.pos_state
    if self.neg_sample != self.pos_sample:
      self.neg_sample.free_device_memory()
      self.neg_sample = self.pos_sample

  def InitializeNegPhase(self, to_pos=False):
    """Initialize negative particles.

    Copies the pos state and samples it to initialize the ngative particles.
    """
    self.SetPhase(pos=False)
    if to_pos:
      self.state.assign(self.pos_state)
    else:
      self.ResetState(rand=True)
    self.Sample()
    self.SetPhase(pos=True)

  def AllocateBatchsizeDependentMemory(self, batchsize):
    super(DBMLayer, self).AllocateBatchsizeDependentMemory(batchsize)

    # self.state and self.deriv were allocated in super but they are not needed
    # for DBMs, so we re-interpret them as:
    self.pos_state = self.state
    self.pos_sample = self.deriv
    self.sample = self.pos_sample

    # Allocate variables for negative state.
    if self.phases_tied:
      self.neg_state = self.pos_state
      self.neg_sample = self.pos_sample
    else:
      self.neg_state = cm.CUDAMatrix(np.zeros((self.numlabels * self.dimensions,
                                              batchsize)))
      self.neg_sample = cm.CUDAMatrix(np.zeros((self.numlabels * self.dimensions,
                                              batchsize)))

  def ComputeUp(self, train=False, recon=False, step=0, maxsteps=0):
    """
    Computes the state of a layer, given the state of its incoming neighbours.

    Args:
      train: True if this computation is happening during training, False during
        evaluation.
      recon: If True, the input layer will be reconstructed from the model and
        the error will be reported. If False, this will not happen.
      step: Training step.
      maxsteps: Maximum number of steps that will be taken (Some hyperparameters
        may depend on this.)
  """
    logging.debug('ComputeUp in %s', self.name)
    if self.is_input and self.pos_phase and not recon:
      self.GetData()
    else:
      for i, edge in enumerate(self.incoming_edge):
        neighbour = self.incoming_neighbour[i]
        if self.pos_phase:
          # Mean field in pos phase
          inputs = neighbour.state
        else:
          # Gibbs sampling in neg phase
          inputs = neighbour.sample
        if edge.node2 == self:
          w = edge.params['weight'].T
          factor = edge.proto.up_factor
        else:
          w = edge.params['weight']
          factor = edge.proto.down_factor
        if i == 0:
          cm.dot(w, inputs, target=self.state)
          if factor != 1:
            self.state.mult(factor)
        else:
          self.state.add_dot(w, inputs, mult=factor)
      b = self.params['bias']
      if self.replicated_neighbour is None:
        self.state.add_col_vec(b)
      else:
        self.state.add_dot(b, self.replicated_neighbour.NN)
      self.ApplyActivation()
    if self.hyperparams.dropout:
      if train and maxsteps - step >= self.hyperparams.stop_dropout_for_last:
        # Randomly set states to zero.
        if self.pos_phase:
          self.mask.fill_with_rand()
          self.mask.greater_than(self.hyperparams.dropout_prob)
        self.state.mult(self.mask)
      else:
        # Produce expected output.
        self.state.mult(1.0 - self.hyperparams.dropout_prob)

  def CollectSufficientStatistics(self):
    """Collect sufficient statistics for this layer."""
    logging.debug('Collecting suff stats %s', self.name)
    h = self.hyperparams

    if self.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.state.div_by_row(self.NN)

    if self.pos_phase:
      self.state.sum(axis=1, target=self.suff_stats)
      if h.sparsity:
        damping = h.sparsity_damping
        self.means.mult(damping)
        self.means.add_mult(self.suff_stats, alpha=(1-damping)/self.batchsize)
        self.means.subtract(h.sparsity_target, target=self.means_temp)
        self.suff_stats.add_mult(self.means_temp,
                                 alpha=-self.batchsize * h.sparsity_cost)
      if self.learn_precision:
        temp = self.deriv
        b = self.params['bias']
        self.state.add_col_mult(b, mult=-1.0, target=temp)
        temp.mult(temp)
        temp.sum(axis=1, target=self.suff_stats2)
    else:
      self.suff_stats.add_sums(self.state, axis=1, mult=-1.0)
      if self.learn_precision:
        temp = self.deriv
        b = self.params['bias']
        self.state.add_col_mult(b, mult=-1.0, target=temp)
        temp.mult(temp)
        self.suff_stats2.add_sums(temp, axis=1, mult=-1.0)

    if self.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.state.mult_by_row(self.NN)

    if self.pos_phase and h.sparsity:
      return float(self.means.asarray().mean())

  def UpdateParams(self, step=0):
    """Update parameters associated with this layer."""
    logging.debug('UpdateParams in %s', self.name)
    h = self.hyperparams
    numcases = self.batchsize

    if h.epsilon_decay == deepnet_pb2.Hyperparams.NONE:
      epsilon = h.base_epsilon
    elif h.epsilon_decay == deepnet_pb2.Hyperparams.INVERSE_T:
      epsilon = h.base_epsilon / (1 + float(step) / h.epsilon_decay_half_life)
    elif h.epsilon_decay == deepnet_pb2.Hyperparams.EXPONENTIAL:
      epsilon = h.base_epsilon / np.power(2, float(step) / h.epsilon_decay_half_life)
    if step < h.start_learning_after:
      epsilon = 0.0

    if h.momentum_change_steps > step:
      f = float(step) / h.momentum_change_steps
      momentum = (1.0 - f) * h.initial_momentum + f * h.final_momentum
    else:
      momentum = h.final_momentum

    # Update bias.
    if self.learn_precision:
      p = self.params['precision']
      p.upper_bound(h.precision_upper_bound)
      self.suff_stats.mult(p)
    b = self.params['bias']
    #b_delta = self.params['grad_bias']
    b_delta = self.grad_bias
    b_delta.mult(momentum)
    b_delta.add_mult(self.suff_stats, 1.0 / numcases)
    if h.apply_l2_decay:
      b_delta.add_mult(b, -h.l2_decay)

    if self.learn_precision:
      p = self.params['precision']
      b.divide(p)
      b.add_mult(b_delta, epsilon)
      b.mult(p)
    else:
      b.add_mult(b_delta, epsilon)

    # Update precision.
    if self.learn_precision:
      p = self.params['precision']
      p_delta = self.params['grad_precision']
      p_delta.assign(0)
      for i, edge in enumerate(self.incoming_edge):
        inputs = edge.suff_stats
        temp = edge.temp
        if edge.node2 == self:
          w = edge.params['weight'].T
        else:
          w = edge.params['weight']
        inputs.mult(w, target=temp)
        p_delta.add_sums(temp, axis=1)
      p_delta.subtract(self.suff_stats2)
      p_delta.divide(p)
      p.add_mult(p_delta, h.precision_epsilon / numcases)
      p.upper_bound(h.precision_upper_bound)
      p.lower_bound(0)

