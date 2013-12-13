"""Implements a layer of neurons."""
from parameter import *
import matplotlib.pyplot as plt
plt.ion()
class Layer(Parameter):

  def __init__(self, proto, t_op=None, tied_to=None):
    super(Layer, self).__init__()
    self.tied_to = tied_to
    if proto.tied:
      tied_to.num_shares += 1
      proto = util.LoadMissing(proto, tied_to.proto)
    self.proto = proto
    self.state = None
    self.params = {}
    self.hyperparams = proto.hyperparams
    self.incoming_edge = []
    self.outgoing_edge = []
    self.outgoing_neighbour = []
    self.incoming_neighbour = []
    self.use_suff_stats = False
    self.fast_dropout_partner = None
    if t_op:
      self.batchsize = t_op.batchsize
      self.use_suff_stats = t_op.optimizer == deepnet_pb2.Operation.PCD \
          or t_op.optimizer == deepnet_pb2.Operation.CD
    else:
      self.batchsize = 0
    self.name = proto.name
    self.dimensions = proto.dimensions
    self.numlabels = proto.numlabels
    self.activation = proto.hyperparams.activation
    self.is_input = proto.is_input
    self.is_output = proto.is_output
    self.loss_function = proto.loss_function
    self.loss_weight = proto.loss_weight
    self.train_data_handler = None
    self.validation_data_handler = None
    self.test_data_handler = None
    self.tied_to = None
    self.data_tied_to = None
    self.data = None
    self.deriv = None
    self.prefix = proto.prefix
    self.marker = 0
    self.fig = visualize.GetFigId()
    self.tiny = 1e-10
    self.replicated_neighbour = None
    self.is_initialized = proto.is_initialized
    self.t_op = t_op
    self.learn_precision = False
    self.sample_input = self.hyperparams.sample_input
    self.LoadParams(proto, t_op=t_op, tied_to=tied_to)
    if self.batchsize > 0:
      self.AllocateMemory(self.batchsize)

  def LoadParams(self, proto, **kwargs):
    assert proto
    for param in proto.param:
      if not param.dimensions:
        param.dimensions.extend([proto.numlabels * proto.dimensions, 1])
      elif len(param.dimensions) == 1:
        param.dimensions.append(1)
    super(Layer, self).LoadParams(proto, **kwargs)

  def LoadPretrained(self, param):
    node_name = param.pretrained_model_node1
    if node_name == '':
      node_name = self.proto.name
    mat = None
    for pretrained_model in param.pretrained_model:
      model_file = os.path.join(self.prefix, pretrained_model)
      ext = os.path.splitext(pretrained_model)[1]
      if ext == '.npz':
        npzfile = np.load(model_file)
        if param.name == 'bias':
          this_mat = np.nan_to_num(npzfile['mean'] / npzfile['std'])
        elif param.name == 'precision':
          this_mat = np.nan_to_num(1. / npzfile['std'])
      elif ext == '.npy':
        this_mat = np.load(model_file)
      else:
        model = util.ReadModel(model_file)
        # Find the relevant node in the model.
        node = next(n for n in model.layer if n.name == node_name)
        # Find the relevant parameter in the node.
        pretrained_param = next(p for p in node.param if p.name == param.name)
        assert pretrained_param.mat != '',\
                'Pretrained param %s in layer %s of model %s is empty!!' % (
                  pretrained_param.name, node.name, pretrained_model)
        this_mat = util.ParameterAsNumpy(pretrained_param)
      if len(this_mat.shape) == 1:
        this_mat = this_mat.reshape(-1, 1)
      if mat is None:
        mat = this_mat
      else:
        mat += this_mat
    return mat / len(param.pretrained_model)

  def SetData(self, data):
    self.data = data

  def AddIncomingEdge(self, edge):
    if edge not in self.incoming_edge:
      self.incoming_edge.append(edge)
      if self == edge.node1:
        neighbour = edge.node2
      else:
        neighbour = edge.node1
      self.incoming_neighbour.append(neighbour)
      if neighbour.proto.replicate_bias and neighbour.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
        self.replicated_neighbour = neighbour

  def AddOutgoingEdge(self, edge):
    if edge not in self.outgoing_edge:
      self.outgoing_edge.append(edge)
      if self == edge.node1:
        self.outgoing_neighbour.append(edge.node2)
      else:
        self.outgoing_neighbour.append(edge.node1)

  def PrintNeighbours(self):
    for n in self.incoming_neighbour:
      print "Incoming edge from %s" % n.name
    for n in self.outgoing_neighbour:
      print "Outgoing edge to %s" % n.name

  def ResetState(self, rand=False):
    if rand:
      self.state.fill_with_randn()
      self.ApplyActivation()
    else:
      self.state.assign(0)

  def GetData(self):
    self.state.assign(self.data)

  def GetSparsityGradient(self):
    h = self.hyperparams
    damping = h.sparsity_damping
    target = h.sparsity_target
    cost = h.sparsity_cost

    # Update \hat{\rho}.
    self.means.mult(damping)
    self.means.add_sums(self.state, axis=1, mult=(1-damping)/self.batchsize)
    
    # Compute gradient.
    self.means.subtract(target, target=self.sparsity_gradient)
    div = self.GetSparsityDivisor()
    self.sparsity_gradient.divide(div)
    self.sparsity_gradient.mult(cost)

    # Return gradient.
    return self.sparsity_gradient

  def AllocateMemory(self, batchsize):
    self.AllocateBatchsizeDependentMemory(batchsize)
    dimensions = self.dimensions
    numlabels = self.numlabels
    numdims = dimensions * numlabels
    self.dimsize = cm.CUDAMatrix(np.zeros((numdims, 1)))
    if self.hyperparams.sparsity:
      tgt = self.hyperparams.sparsity_target
      self.means = cm.CUDAMatrix(tgt + np.zeros((numdims, 1)))
      self.sparsity_gradient = cm.CUDAMatrix(np.zeros((numdims, 1)))
      self.means_temp2 = cm.CUDAMatrix(np.zeros((numdims, 1)))
    self.gradient = cm.CUDAMatrix(np.zeros((numdims, 1)))
    self.gradient_history = cm.CUDAMatrix(np.zeros((numdims, 1)))

  def AllocateBatchsizeDependentMemory(self, batchsize):
    if self.data:
      self.data.free_device_memory()
    if self.deriv:
      self.deriv.free_device_memory()
    self.batchsize = batchsize
    dimensions = self.dimensions
    numlabels = self.numlabels
    numdims = dimensions * numlabels
    self.statesize = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
    self.batchsize_temp = cm.CUDAMatrix(np.zeros((1, batchsize)))
    self.state = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
    self.deriv = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
    if self.t_op:
      if self.t_op.optimizer == deepnet_pb2.Operation.PCD:
        self.pos_state = self.state
        self.pos_sample = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.neg_state = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.neg_sample = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.sample = self.pos_sample
        self.suff_stats = cm.empty((numdims, 1))
      elif self.t_op.optimizer == deepnet_pb2.Operation.CD:
        self.sample = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.suff_stats = cm.empty((numdims, 1))
    else:
        self.state = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
    if self.is_input or self.is_initialized or self.is_output:
      self.data = cm.CUDAMatrix(np.zeros((dimensions, batchsize)))
    if self.hyperparams.dropout:
      self.mask = cm.CUDAMatrix(np.zeros(self.state.shape))

  def CollectSufficientStatistics(self, neg=False):
    """Collect sufficient statistics for this layer."""
    h = self.hyperparams
    if not neg:
      self.state.sum(axis=1, target=self.suff_stats)
      if h.sparsity:
        sparsity_gradient = self.GetSparsityGradient()
        self.suff_stats.add_mult(sparsity_gradient, -self.batchsize)
    else:
      self.suff_stats.add_sums(self.state, axis=1, mult=-1.0)
    if not neg and h.sparsity:
      return self.means.sum()/self.means.shape[0]

  def Show(self, train=False):
    """Displays useful statistics about the model."""
    if not self.proto.hyperparams.enable_display:
      return
    f = 1
    if self.hyperparams.dropout and not train:
      f = 1 / (1 - self.hyperparams.dropout_prob)
    if self.is_input:
      visualize.display_hidden(self.data.asarray(), self.fig, title=self.name)
      #visualize.display_w(self.neg_sample.asarray(), 28, 10, self.state.shape[1]/10, self.fig, title=self.name, vmax=1, vmin=0)
      #visualize.show_hist(self.params['bias'].asarray(), self.fig)
    else:
      visualize.display_hidden(f*self.state.asarray(), self.fig, title=self.name)
      #visualize.show_hist(self.params['bias'].asarray(), self.fig)
      """
      plt.figure(self.fig)
      plt.clf()
      plt.subplot(1, 3, 1)
      plt.title('pos_probabilities')
      plt.imshow(self.pos_state.asarray(), cmap = plt.cm.gray, interpolation = 'nearest', vmax=1, vmin=0)
      plt.subplot(1, 3, 2)
      plt.title('neg_probabilities')
      plt.imshow(self.neg_state.asarray(), cmap = plt.cm.gray, interpolation = 'nearest', vmax=1, vmin=0)
      plt.subplot(1, 3, 3)
      plt.title('neg_samples')
      plt.imshow(self.neg_sample.asarray(), cmap = plt.cm.gray, interpolation = 'nearest', vmax=1, vmin=0)
      plt.suptitle(self.name)
      plt.draw()
      """
      #visualize.display_w(self.neg_sample.asarray(), 1, 1, self.state.shape[1], self.fig, title=self.name)

def display_w(w, s, r, c, fig, vmax=None, vmin=None, dataset='mnist', title='weights'):

  def ComputeDeriv(self):
    pass
  def GetLoss(self, get_deriv=False):
    pass
  def Sample(self):
    pass
  def ApplyActivation(self):
    pass
  def GetSparsityDivisor(self):
    self.means_temp2.assign(1)
    return self.means_temp2
