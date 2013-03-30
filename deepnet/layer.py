"""Implements a layer of neurons."""
import cudamat as cm
import deepnet_pb2
import logging
import numpy as np
import os.path
import util
import visualize
import pdb

class Layer(object):
  id_counter = 0

  def __init__(self, proto, t_op=None):
    self.proto = proto
    self.id = Layer.id_counter
    Layer.id_counter += 1
    self.state = None
    self.params = {}
    self.hyperparams = None
    self.incoming_edge = []
    self.outgoing_edge = []
    self.outgoing_neighbour = []
    self.incoming_neighbour = []
    self.use_suff_stats = False
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
    self.train_data_handler = None
    self.validation_data_handler = None
    self.test_data_handler = None
    self.tied_to = None
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
    self.sample_input = False
    self.LoadParams(proto)
    if self.batchsize > 0:
      self.AllocateMemory(self.batchsize)

  def SaveParameters(self):
    for param in self.proto.param:
      param.mat = util.NumpyAsParameter(self.params[param.name].asarray())

  def SetData(self, data):
    self.data = data

  def InitializeParameter(self, param):
    if param.initialization == deepnet_pb2.Parameter.CONSTANT:
      return np.zeros(tuple(param.dimensions)) + param.constant
    elif param.initialization == deepnet_pb2.Parameter.DENSE_GAUSSIAN:
      return param.sigma * np.random.randn(*tuple(param.dimensions))
    elif param.initialization == deepnet_pb2.Parameter.PRETRAINED:
      node_name = param.pretrained_model_node1
      if node_name == '':
        node_name = self.proto.name
      mat = None
      for pretrained_model in param.pretrained_model:
        if os.path.splitext(pretrained_model)[1] == '.npz':
          model_file = os.path.join(self.prefix, pretrained_model)
          npzfile = np.load(model_file)
          if param.name == 'bias':
            this_mat = np.nan_to_num(npzfile['mean'] / npzfile['std'])
          elif param.name == 'precision':
            this_mat = np.nan_to_num(1. / npzfile['std'])
        else:
          model_file = os.path.join(self.prefix, pretrained_model)
          model = util.ReadModel(model_file)
          # Find the relevant node in the model.
          node = next(n for n in model.layer if n.name == node_name)
          # Find the relevant parameter in the node.
          pretrained_param = next(p for p in node.param if p.name == param.name)
          assert pretrained_param.mat != '',\
                  'Pretrained param %s in layer %s of model %s is empty!!' % (
                    pretrained_param.name, node.name, pretrained_model)
          this_mat = util.ParameterAsNumpy(pretrained_param)
        if mat is None:
          mat = this_mat
        else:
          mat += this_mat
      return mat / len(param.pretrained_model)
    else:
      raise Exception('Unknown parameter initialization.')

  def LoadParams(self, proto):
    self.hyperparams = proto.hyperparams
    param_names = [param.name for param in proto.param]
    for param in proto.param:
      if not param.dimensions:
        param.dimensions.extend([proto.numlabels * proto.dimensions])
      if param.mat:
        mat = util.ParameterAsNumpy(param).reshape(-1, 1)
      else:
        mat = self.InitializeParameter(param).reshape(-1, 1)
      self.params[param.name] = cm.CUDAMatrix(mat)
      if param.name == 'bias':
        self.grad_bias = cm.empty(mat.shape)
        self.grad_bias.assign(0)
    self.sample_input = self.hyperparams.sample_input

  def AddIncomingEdge(self, edge):
    if edge not in self.incoming_edge:
      self.incoming_edge.append(edge)
      if self == edge.node1:
        neighbour = edge.node2
      else:
        neighbour = edge.node1
      self.incoming_neighbour.append(neighbour)
      if neighbour.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX and \
         neighbour.proto.replicate_bias:
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
      print "Incoming edge from %d (%s)" % (n.id, n.name)
    for n in self.outgoing_neighbour:
      print "Outgoing edge to %d (%s)" % (n.id, n.name)

  def ResetState(self, rand=False):
    if rand:
      self.state.fill_with_randn()
      self.ApplyActivation()
    else:
      self.state.assign(0)

  def GetData(self):
    self.state.assign(self.data)

  def GetMomentumAndEpsilon(self, step):
    h = self.hyperparams

    # Linearly interpolate between initial and final momentum.
    if h.momentum_change_steps > step:
      f = float(step) / h.momentum_change_steps
      momentum = (1.0 - f) * h.initial_momentum + f * h.final_momentum
    else:
      momentum = h.final_momentum

    # Decide learning rate.
    if h.epsilon_decay == deepnet_pb2.Hyperparams.NONE:
      epsilon = h.base_epsilon
    elif h.epsilon_decay == deepnet_pb2.Hyperparams.INVERSE_T:
      epsilon = h.base_epsilon / (1 + float(step) / h.epsilon_decay_half_life)
    elif h.epsilon_decay == deepnet_pb2.Hyperparams.EXPONENTIAL:
      epsilon = h.base_epsilon / np.power(2, float(step) / h.epsilon_decay_half_life)
    if step < h.start_learning_after:
      epsilon = 0.0
    return momentum, epsilon

  def AddSparsityGradient(self):
    h = self.hyperparams
    damping = h.sparsity_damping
    target = h.sparsity_target
    cost = h.sparsity_cost

    # Update \hat{\rho}.
    self.means.mult(damping)
    self.means.add_sums(self.state, axis=1, mult=(1-damping)/self.batchsize)

    # Compute gradient.
    div = self.GetSparsityDivisor()

    self.means.subtract(target, target=self.means_temp)
    self.means_temp.divide(self.means_temp2)
    self.means_temp.mult(cost)

    # Add to the derivative of the loss.
    if self.use_suff_stats:
      self.suff_stats.add_mult(self.means_temp, alpha=-self.batchsize)
    else:
      self.deriv.add_col_vec(self.means_temp)

  def AllocateMemory(self, batchsize):
    self.AllocateBatchsizeDependentMemory(batchsize)
    dimensions = self.dimensions
    numlabels = self.numlabels
    numdims = dimensions * numlabels
    self.dimsize = cm.CUDAMatrix(np.zeros((numdims, 1)))
    self.unitcell = cm.CUDAMatrix(np.zeros((1,1)))
    if self.hyperparams.sparsity:
      tgt = self.hyperparams.sparsity_target
      self.means = cm.CUDAMatrix(tgt + np.zeros((numdims, 1)))
      self.means_temp = cm.CUDAMatrix(np.zeros((numdims, 1)))
      self.means_temp2 = cm.CUDAMatrix(np.zeros((numdims, 1)))

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
    if self.t_op:
      if self.t_op.optimizer == deepnet_pb2.Operation.PCD:
        self.pos_state = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.pos_sample = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.neg_state = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.neg_sample = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.state = self.pos_state
        self.sample = self.pos_sample
        self.suff_stats = cm.empty((numdims, 1))
      elif self.t_op.optimizer == deepnet_pb2.Operation.CD:
        self.state = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.sample = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.suff_stats = cm.empty((numdims, 1))
      else:
        self.state = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
        self.deriv = cm.CUDAMatrix(np.zeros((numdims, batchsize)))
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
        self.AddSparsityGradient()
    else:
      self.suff_stats.add_sums(self.state, axis=1, mult=-1.0)
    if not neg and h.sparsity:
      return float(self.means.asarray().mean())

  def Show(self, train=False):
    """Displays useful statistics about the model."""
    if not self.proto.hyperparams.enable_display:
      return
    f = 1
    if self.hyperparams.dropout and not train:
      f = 1 / (1 - self.hyperparams.dropout_prob)
    if self.is_input:
      visualize.display_hidden(self.data.asarray(), self.fig, title=self.name)
    else:
      visualize.display_hidden(f*self.state.asarray(), self.fig, title=self.name)

  def ComputeDeriv(self):
    pass
  def GetLoss(self, get_deriv=False):
    pass
  def Sample(self):
    pass
  def ApplyActivation(self):
    pass



