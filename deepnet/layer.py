"""Implements a layer of neurons."""
import cudamat as cm
import deepnet_pb2
import logging
import numpy as np
import util
import visualize

from cudamat_conv import cudamat_conv2 as cc

class Layer(object):
  id_counter = 0

  def __init__(self, proto, batchsize=0):
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
    self.batchsize = batchsize
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
    self.suff_stats = None
    self.LoadParams(proto)
    self.marker = 0
    self.fig = visualize.GetFigId()
    self.fig_neg = visualize.GetFigId()
    self.pos_phase = True
    self.tiny = 1e-10
    self.replicated_neighbour = None
    if batchsize > 0:
      self.AllocateMemory(batchsize)

  def SaveParameters(self):
    for param in self.proto.param:
      param.mat = util.NumpyAsParameter(self.params[param.name].asarray())

  def SetDataHandles(self, train=None, valid=None, test=None, tied_to=None):
    if tied_to is None:
      self.train_data_handler = train
      self.validation_data_handler = valid
      self.test_data_handler = test
    else:
      self.tied_to = tied_to

  def GetTestData(self):
    if self.tied_to is None:
      self.data = self.test_data_handler.Get()
    else:
      self.data.assign(self.tied_to.data)

  def GetValidationData(self):
    if self.tied_to is None:
      self.data = self.validation_data_handler.Get()
    else:
      self.data.assign(self.tied_to.data)

  def GetTrainData(self):
    if self.tied_to is None:
      self.data = self.train_data_handler.Get()
    else:
      self.data.assign(self.tied_to.data)

  def Show(self):
    if not self.proto.hyperparams.enable_display:
      return
    """
    if self.is_input and hasattr(self, 'neg_state'):
      visualize.display_w(self.neg_state.asarray(), self.proto.shape[0],
                          10, self.batchsize/10, self.fig, title='neg particles')
    elif self.is_input:
      if len(self.proto.shape) == 3:
        edge = self.outgoing_edge[0]
        visualize.display_convw(self.state.asarray().T,
                                self.proto.shape[0],
                                16, self.batchsize/16, self.fig,
                                title=self.name)

      else:
        visualize.display_w(self.state.asarray(), self.proto.shape[0],
                            10, self.batchsize/10, self.fig, title='data')
    else:
    """
    if self.is_input:
      visualize.display_hidden(self.data.asarray(), self.fig, title=self.name)
    else:
      visualize.display_hidden(self.pos_state.asarray(), 2*self.fig, title=self.name + "_positive")
      visualize.display_hidden(self.neg_state.asarray(), 2*self.fig_neg, title=self.name + "_negative")
      #visualize.display_hidden(self.params['bias'].asarray(), 2*self.fig_neg, title=self.name + "_bias")
      visualize.display_w(self.pos_state.asarray(), self.proto.shape[0],
                          self.batchsize, 1, 2*self.fig+1,
                          title=self.name + "_positive", vmin=0, vmax=1)
      visualize.display_w(self.neg_sample.asarray(), self.proto.shape[0],
                          self.batchsize, 1, 2*self.fig_neg+1,
                          title=self.name + "_negative", vmin=0, vmax=1)
      """
      try:
        visualize.display_hidden(self.neg_state.asarray(), self.fig_neg, title=self.name + "_negative")
      except Exception as e:
        print 'Problem displaying %s_negative' % self.name
      """

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
        model = util.ReadModel(pretrained_model)
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
      if 'grad_'+param.name not in param_names and not param.name.startswith(
        'grad_'):
        grad_p = deepnet_pb2.Parameter()
        grad_p.name = 'grad_'+param.name
        proto.param.extend([grad_p])

    for param in proto.param:
      if not param.dimensions:
         param.dimensions.extend([proto.numlabels * proto.dimensions])
      if param.mat and 'grad' not in param.name:
        mat = util.ParameterAsNumpy(param).reshape(-1, 1)
      else:
        mat = self.InitializeParameter(param).reshape(-1, 1)
      self.params[param.name] = cm.CUDAMatrix(mat)

  def AddIncomingEdge(self, edge):
    if edge not in self.incoming_edge:
      self.incoming_edge.append(edge)
      if self == edge.node1:
        neighbour = edge.node2
      else:
        neighbour = edge.node1
      self.incoming_neighbour.append(neighbour)
      if neighbour.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
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

  def AllocateMemory(self, batchsize):
    if self.data:
      self.data.free_device_memory()
    if self.deriv:
      self.deriv.free_device_memory()
    if self.is_input or self.is_output:
      self.data = cm.CUDAMatrix(np.zeros((self.dimensions, batchsize)))
    self.deriv = cm.CUDAMatrix(np.zeros((self.numlabels * self.dimensions,
                                         batchsize)))
    self.state = cm.CUDAMatrix(np.zeros((self.numlabels * self.dimensions,
                                         batchsize)))
    self.temp = cm.CUDAMatrix(np.zeros((self.dimensions, batchsize)))

    if self.hyperparams.sparsity:
      self.means = cm.CUDAMatrix(0.5 + np.zeros(
        (self.numlabels * self.dimensions, 1)))
      self.means_temp = cm.CUDAMatrix(np.zeros(
        (self.numlabels * self.dimensions, 1)))

    if self.activation == deepnet_pb2.Hyperparams.SOFTMAX or\
       self.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.expansion_matrix = cm.CUDAMatrix(np.eye(self.numlabels))
      self.expanded_batch = cm.CUDAMatrix(np.zeros(
        (self.numlabels * self.dimensions, batchsize)))
    if self.loss_function == deepnet_pb2.Layer.CROSS_ENTROPY:
      self.unitcell = cm.CUDAMatrix(np.zeros((1,1)))
      if self.activation == deepnet_pb2.Hyperparams.SOFTMAX:
        self.temp2 = cm.CUDAMatrix(np.zeros((self.dimensions, batchsize)))
        self.indices = cm.CUDAMatrix(np.zeros((1,self.dimensions * batchsize)))
        self.rowshift = cm.CUDAMatrix(
          self.numlabels*np.arange(self.dimensions * batchsize).reshape(1, -1))
      elif self.activation == deepnet_pb2.Hyperparams.LOGISTIC:
        self.temp3 = cm.CUDAMatrix(np.zeros((self.dimensions, 1)))
    if self.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.NN = cm.CUDAMatrix(self.hyperparams.normalize_to + np.zeros((1, batchsize)))
    if self.hyperparams.dropout:
      self.mask = cm.CUDAMatrix(np.zeros(self.state.shape))

  def ResetState(self, rand=False):
    if self.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      if self.hyperparams.normalize:
        self.NN.assign(self.hyperparams.normalize_to)
      else:
        self.NN.assign(1)
    if rand:
      self.state.fill_with_randn()
      self.ApplyActivation()
    else:
      self.state.assign(0)

  def Sample(self):
    logging.debug('Sample in %s', self.name)
    state = self.state
    sample = self.sample
    if self.activation == deepnet_pb2.Hyperparams.LOGISTIC:
      state.sample_binomial(target=sample)
    elif self.activation == deepnet_pb2.Hyperparams.TANH:
      state.sample_binomial_tanh(target=sample)
    elif self.activation == deepnet_pb2.Hyperparams.RECTIFIED_LINEAR:
      state.sample_poisson(target=sample)
    elif self.activation == deepnet_pb2.Hyperparams.RECTIFIED_LINEAR_SMOOTH:
      state.sample_poisson(target=sample)
    elif self.activation == deepnet_pb2.Hyperparams.LINEAR:
      #sample.assign(state)
      state.sample_gaussian(target=sample, mult=0.01)
    elif self.activation == deepnet_pb2.Hyperparams.SOFTMAX:
      sample.fill_with_rand()
      cm.log(sample)
      sample.mult(-1)
      sample.reciprocal()
      sample.mult(state)
      sample.argmax(axis=0, target=self.temp)
      self.expansion_matrix.select_columns(self.temp, target=sample)
    elif self.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      if self.proto.hyperparams.normalize:
        sample.assign(0)
        temp_sample = self.expanded_batch
        numsamples = int(self.proto.hyperparams.normalize_to)
        for i in range(numsamples):
          temp_sample.fill_with_rand()
          cm.log(temp_sample)
          temp_sample.mult(-1)
          state.divide(temp_sample, target=temp_sample)
          temp_sample.max(axis=0, target=self.temp)
          temp_sample.add_row_mult(self.temp, -1)
          temp_sample.less_than(0)
          sample.add(temp_sample)
        sample.subtract(numsamples)
        sample.mult(-1)
      else:  # This is an approximation.
        sample.fill_with_rand()
        cm.log(sample)
        sample.mult(-1)
        sample.reciprocal()
        sample.mult(state)
        sample.argmax(axis=0, target=self.temp)
        self.expansion_matrix.select_columns(self.temp, target=sample)
        sample.mult_by_row(self.NN)
    else:
      raise Exception('Unknown activation')

  def ApplyActivation(self):
    state = self.state
    if self.activation == deepnet_pb2.Hyperparams.LOGISTIC:
      cm.sigmoid(state)
    elif self.activation == deepnet_pb2.Hyperparams.TANH:
      cm.tanh(state)
    elif self.activation == deepnet_pb2.Hyperparams.RECTIFIED_LINEAR:
      state.greater_than(0, target=self.temp)
      state.mult(self.temp)
    elif self.activation == deepnet_pb2.Hyperparams.RECTIFIED_LINEAR_SMOOTH:
      cm.log_1_plus_exp(state)
    elif self.activation == deepnet_pb2.Hyperparams.LINEAR:
      pass
    elif self.activation == deepnet_pb2.Hyperparams.SOFTMAX:
      state.max(axis=0, target=self.temp)
      state.add_row_mult(self.temp, -1)
      cm.exp(state)
      state.sum(axis=0, target=self.temp)
      self.temp.reciprocal()
      state.mult_by_row(self.temp)
    elif self.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      state.max(axis=0, target=self.temp)
      state.add_row_mult(self.temp, -1)
      cm.exp(state)
      state.sum(axis=0, target=self.temp)
      self.NN.divide(self.temp, target=self.temp)
      state.mult_by_row(self.temp)
    else:
      raise Exception('Unknown activation')

  def ConvolveUp(self, inputs, edge, target):
    w = edge.params['weight']
    conv = edge.conv_params
    size = conv.size
    stride = conv.stride
    padding = conv.padding
    num_filters = conv.num_filters
    num_colors = conv.num_colors

    f, numdims = w.shape
    assert f == num_filters, 'f is %d but num_filters is %d' % (f, num_filters)
    assert numdims == size**2 * num_colors

    input_t = edge.input_t
    if conv.max_pool:
      output_t = edge.unpooled_layer
    else:
      output_t = edge.output_t
    numimages, numdims = input_t.shape
    numimages2, numdims2 = output_t.shape

    assert numimages == numimages2, '%d %d' % (numimages, numimages2)

    assert numdims % num_colors == 0
    x = int(np.sqrt(numdims / num_colors))
    assert x**2 == numdims/num_colors

    n_locs = (x + 2 * padding - size) / stride + 1

    inputs.transpose(input_t)
    cc.convUp(input_t, w, output_t, n_locs, padding, stride, num_colors)

    # Do maxpooling
    if conv.max_pool:
      n_locs = (n_locs + 2 * padding - conv.pool_size) / conv.pool_stride + 1
      cc.MaxPool(output_t, edge.output_t, num_filters, conv.pool_size, 0, conv.pool_stride, n_locs)
      output_t = edge.output_t
    output_t.transpose(target)


  def AddConvolveUp(self, inputs, edge, target):
    raise Exception('Not implemented.')

  def GetData(self):
    if self.activation == deepnet_pb2.Hyperparams.SOFTMAX:
      self.expansion_matrix.select_columns(self.data, target=self.state)
    else:
      self.state.assign(self.data)
    if self.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.state.sum(axis=0, target=self.NN)
      if self.hyperparams.normalize:  # normalize word count vector.
        self.NN.add(self.tiny)
        self.state.div_by_row(self.NN)
        self.state.mult(self.hyperparams.normalize_to)
        self.NN.assign(self.hyperparams.normalize_to)

  def ComputeUp(self, train=False, step=0, maxsteps=0):
    """
    Computes the state of a layer, given the state of its incoming neighbours.

    Args:
      train: True if this computation is happening during training, False during
        evaluation.
      step: Training step.
      maxsteps: Maximum number of steps that will be taken (Some hyperparameters
        may depend on this.)
    """
    logging.debug('ComputeUp in %s', self.name)
    self.dirty = False
    if self.is_input:
      self.GetData()
    else:
      for i, edge in enumerate(self.incoming_edge):
        if edge in self.outgoing_edge:
          continue
        inputs = self.incoming_neighbour[i].state
        if edge.conv:
          if i == 0:
            self.ConvolveUp(inputs, edge, self.state)
          else:
            self.AddConvoleUp(inputs, edge, self.state)
        else:
          w = edge.params['weight']
          factor = edge.proto.up_factor
          if i == 0:
            cm.dot(w.T, inputs, target=self.state)
            if factor != 1:
              self.state.mult(factor)
          else:
            self.state.add_dot(w.T, inputs, mult=factor)
      b = self.params['bias']
      if self.replicated_neighbour is None:
        self.state.add_col_vec(b)
      else:
        self.state.add_dot(b, self.replicated_neighbour.NN)
      self.ApplyActivation()

    if self.hyperparams.dropout:
      if train and maxsteps - step >= self.hyperparams.stop_dropout_for_last:
        # Randomly set states to zero.
        self.mask.fill_with_rand()
        self.mask.greater_than(self.hyperparams.dropout_prob)
        self.state.mult(self.mask)
      else:
        # Produce expected output.
        self.state.mult(1.0 - self.hyperparams.dropout_prob)

  def ComputeDown(self, step):
    """Backpropagate through this layer.
    Args:
      step: The training step. Needed because some hyperparameters depend on
      which training step they are being used in.
    """
    logging.debug('ComputeDown in %s', self.name)
    if self.is_input:  # Nobody to backprop to.
      return
    # At this point self.deriv contains the derivative with respect to the
    # outputs of this layer. Compute derivative with respect to the inputs.
    if self.is_output:
      loss = self.GetLoss(get_deriv=True)
    else:
      self.ComputeDeriv()
      loss = None
    # Now self.deriv contains the derivative w.r.t to the inputs.
    # Send it down each incoming edge and update parameters on the edge.
    for edge in self.incoming_edge:
      if edge.conv:
        edge.node1.AccumulateConvDeriv(edge, self.deriv)
      else:
        edge.node1.AccumulateDeriv(edge, self.deriv)
      edge.UpdateParams(self.deriv, step)
    # Update the parameters on this layer (i.e., the bias).
    self.UpdateParams(self.deriv, step)
    return loss

  def UpdateParams(self, deriv, step):
    """ Update the parameters associated with this layer.

    Update the bias.
    Args:
      deriv: Gradient w.r.t the inputs to this layer.
      step: Training step.
    """
    logging.debug('UpdateParams in %s', self.name)
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
      epsilon = h.base_epsilon / np.pow(2, float(step) / h.epsilon_decay_half_life)
    if step < h.start_learning_after:
      epsilon = 0.0

    b_delta = self.params['grad_bias']
    b = self.params['bias']

    # Update bias.
    b_delta.mult(momentum)
    b_delta.add_sums(deriv, axis=1, mult = (1.0 - momentum) / self.batchsize)
    if h.apply_l2_decay:
      b_delta.add_mult(b, (1-momentum) * h.l2_decay)
    b.add_mult(b_delta, -epsilon)

  def AccumulateConvDeriv(self, edge, deriv):
    """Accumulate the derivative w.r.t the outputs of this layer.

    Each layer needs to compute derivatives w.r.t its outputs. These outputs may
    have been connected to lots of other nodes through outgoing edges.
    This method adds up the derivatives contributed by each outgoing edge.
    It gets derivatives w.r.t the inputs at the other end of an outgoing edge.
    Args:
      edge: The edge which is sending the derivative.
      deriv: The derivative w.r.t the inputs at the other end of this edge.
    """

    if self.dirty:  # If some derivatives have already been received.
      raise Exception('Not implemented.')
    self.dirty = True
    w = edge.params['weight']
    conv = edge.conv_params
    size = conv.size
    stride = conv.stride
    padding = conv.padding
    num_filters = conv.num_filters
    num_colors = conv.num_colors

    f, numdims = w.shape
    assert f == num_filters, 'f is %d but num_filters is %d' % (f, num_filters)
    assert numdims == size**2 * num_colors

    input_t = edge.input_t
    numimages, numdims = input_t.shape

    assert numdims % num_colors == 0
    x = int(np.sqrt(numdims / num_colors))
    assert x**2 == numdims/num_colors

    n_locs = (x + 2 * padding - size) / stride + 1

    if conv.max_pool:
      deriv.transpose(edge.output_t2)
      n_pool_locs = (n_locs + 2 * padding - conv.pool_size) / conv.pool_stride + 1
      cc.MaxPoolUndo(edge.unpooled_layer, edge.unpooled_layer, edge.output_t2,
                     edge.output_t, conv.pool_size, 0, conv.pool_stride, n_pool_locs)
    else:
      deriv.transpose(edge.output_t)

    if self.is_input:
      return
    if conv.max_pool:
      output_t = edge.unpooled_layer
    else:
      output_t = edge.output_t
    cc.convDown(output_t, w, input_t, n_locs, stride, size, x, num_colors)
    input_t.transpose(self.deriv)

  def AccumulateDeriv(self, edge, deriv):
    """Accumulate the derivative w.r.t the outputs of this layer.

    A layer needs to compute derivatives w.r.t its outputs. These outputs may
    have been connected to lots of other nodes through outgoing edges.
    This method adds up the derivatives contributed by each outgoing edge.
    It gets derivatives w.r.t the inputs at the other end of its outgoing edge.
    Args:
      edge: The edge which is sending the derivative.
      deriv: The derivative w.r.t the inputs at the other end of this edge.
    """
    if self.is_input:
      return
    if self.dirty:  # If some derivatives have already been received.
      self.deriv.add_dot(edge.params['weight'], deriv)
    else:  # Receiving derivative for the first time.
      cm.dot(edge.params['weight'], deriv, target=self.deriv)
      self.dirty = True

  def ComputeDeriv(self):
    """Compute derivative w.r.t input given derivative w.r.t output."""
    if self.activation == deepnet_pb2.Hyperparams.LOGISTIC:
      self.deriv.apply_logistic_deriv(self.state)
    elif self.activation == deepnet_pb2.Hyperparams.TANH:
      self.deriv.apply_tanh_deriv(self.state)
      if self.hyperparams.dropout:
        self.deriv.mult(self.mask)
    elif self.activation == deepnet_pb2.Hyperparams.RECTIFIED_LINEAR:
      self.deriv.apply_rectified_linear_deriv(self.state)
    elif self.activation == deepnet_pb2.Hyperparams.RECTIFIED_LINEAR_SMOOTH:
      self.deriv.apply_rectified_linear_smooth_deriv(self.state)
    elif self.activation == deepnet_pb2.Hyperparams.LINEAR:
      if self.hyperparams.dropout:
        self.deriv.mult(self.mask)
    elif self.activation == deepnet_pb2.Hyperparams.SOFTMAX:
      raise Exception('Not implemented.')
    else:
      raise Exception('Unknown activation.')

  def GetLoss(self, get_deriv=False):
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
    if self.loss_function == deepnet_pb2.Layer.CROSS_ENTROPY:
      if self.activation == deepnet_pb2.Hyperparams.LOGISTIC:
        data = self.data
        state = self.state
        deriv = self.deriv
        temp3 = self.temp3
        unitcell = self.unitcell
 
        cm.cross_entropy(data, state, target=deriv, tiny=self.tiny)
        deriv.sum(axis=1, target=temp3)
        temp3.sum(axis=0, target=unitcell)
        cross_entropy = unitcell.euclid_norm()

        cm.correct_preds(data, state, target=deriv, cutoff=0.5)
        deriv.sum(axis=1, target=temp3)
        temp3.sum(axis=0, target=unitcell)
        correct_preds = unitcell.euclid_norm()

        if get_deriv:
          self.state.subtract(self.data, target=self.deriv)

        perf.cross_entropy = cross_entropy
        perf.correct_preds = correct_preds
      elif self.activation == deepnet_pb2.Hyperparams.SOFTMAX:
        temp2 = self.temp2
        temp = self.temp
        batchsize = self.batchsize
        dimensions = self.dimensions
        numlabels = self.numlabels
        state = self.state
        data = self.data
        unitcell = self.unitcell
        indices = self.indices

        # Optimized for space to handle large number of labels in a softmax.
        data.reshape((1, batchsize * dimensions))
        data.add(self.rowshift, target=indices)
        state.reshape((numlabels, dimensions * batchsize))
        state.max(axis=0, target=temp2)
        state.reshape((1, batchsize * numlabels * dimensions))
        state.select_columns(indices, temp)
        temp2.subtract(temp)
        temp2.sign(target=temp2)
        temp2.sum(axis=1, target=unitcell)
        correct_preds = batchsize - unitcell.euclid_norm()
        if get_deriv:
          temp.subtract(1, target=temp2)
          state.set_selected_columns(indices, temp2)
          state.reshape((numlabels * dimensions, batchsize))
          self.deriv.assign(self.state)
        state.reshape((numlabels * dimensions, batchsize))
        temp.add(tiny)
        cm.log(temp)
        temp.sum(axis=1, target=unitcell)
        cross_entropy = unitcell.euclid_norm()
        perf.cross_entropy = cross_entropy
        perf.correct_preds = correct_preds
    elif self.loss_function == deepnet_pb2.Layer.SQUARED_LOSS:
      if self.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX and self.hyperparams.normalize:
        self.data.sum(axis=0, target=self.temp)
        self.temp.add(self.tiny)
        self.data.div_by_row(self.temp, target=self.deriv)
        self.deriv.mult(self.proto.hyperparams.normalize_to)
        self.deriv.subtract(self.state)
      elif self.activation == deepnet_pb2.Hyperparams.SOFTMAX:
        self.expansion_matrix.select_columns(self.data, target=self.expanded_batch)
        self.state.subtract(self.expanded_batch, target=self.deriv)
      else:
        self.state.subtract(self.data, target=self.deriv)
      error = self.deriv.euclid_norm()**2
      perf.error = error
      if self.activation != deepnet_pb2.Hyperparams.SOFTMAX and \
         self.activation != deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
        self.ComputeDeriv()
    else:
      raise Exception('Unknown loss function.')
    return perf
