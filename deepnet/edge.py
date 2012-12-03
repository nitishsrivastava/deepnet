"""Implements an edge connecting two layers of neurons."""
import cudamat as cm
from cudamat_conv import cudamat_conv2 as cc
import numpy as np
import deepnet_pb2
import visualize
import logging
import util

class Edge(object):
  def __init__(self, proto, node1, node2):
    self.proto = proto
    self.id = None
    self.params = {}
    self.hyperparams = None
    self.proto = proto
    self.node1 = node1
    self.node2 = node2
    self.conv = False
    self.name = '%s:%s' % (self.node1.name, self.node2.name)
    if proto.directed:
      node1.AddOutgoingEdge(self)
      node2.AddIncomingEdge(self)
    else:
      node1.AddOutgoingEdge(self)
      node1.AddIncomingEdge(self)
      node2.AddOutgoingEdge(self)
      node2.AddIncomingEdge(self)
    self.LoadParams(proto, node1, node2)
    self.marker = 0
    self.fig = visualize.GetFigId()
    if self.conv:
      self.conv_filter_fig = visualize.GetFigId()

  def Show(self):
    if not self.proto.hyperparams.enable_display:
      return
    if self.node1.is_input:
      if self.conv:
        visualize.display_convw(self.params['weight'].asarray(),
                                self.proto.receptive_field_width,
                                self.proto.display_rows,
                                self.proto.display_cols, self.conv_filter_fig,
                                title=self.name)
        visualize.display_hidden(self.params['weight'].asarray(),
                                 self.fig,
                                 title=self.name)
      else:
        if len(self.node1.proto.shape) < 3:
          visualize.display_wsorted(self.params['weight'].asarray(),
                                    self.proto.receptive_field_width,
                                    self.proto.display_rows,
                                    self.proto.display_cols, self.fig,
                                    title=self.name)
        else:
          visualize.display_convw(self.params['weight'].asarray().T,
                                  self.proto.receptive_field_width,
                                  self.proto.display_rows,
                                  self.proto.display_cols, self.fig,
                                  title=self.name)

  def InitializeParameter(self, param):
    if param.initialization == deepnet_pb2.Parameter.CONSTANT:
      return np.zeros(tuple(param.dimensions)) + param.constant
    elif param.initialization == deepnet_pb2.Parameter.DENSE_GAUSSIAN:
      return param.sigma * np.random.randn(*tuple(param.dimensions))
    elif param.initialization == deepnet_pb2.Parameter.DENSE_UNIFORM:
      return param.sigma * (2 * np.random.rand(*tuple(param.dimensions)) - 1)
    elif param.initialization == deepnet_pb2.Parameter.DENSE_GAUSSIAN_SQRT_FAN_IN:
      if param.conv:
        fan_in = np.prod(param.dimensions[0])
      else:
        fan_in = np.prod(param.dimensions[1])
      stddev = param.sigma / np.sqrt(fan_in)
      return stddev * np.random.randn(*tuple(param.dimensions))
    elif param.initialization == deepnet_pb2.Parameter.DENSE_UNIFORM_SQRT_FAN_IN:
      if param.conv:
        fan_in = np.prod(param.dimensions[0])
      else:
        fan_in = np.prod(param.dimensions[1])
      stddev = param.sigma / np.sqrt(fan_in)
      return stddev * (2 * np.random.rand(*tuple(param.dimensions)) - 1)
    elif param.initialization == deepnet_pb2.Parameter.PRETRAINED:
      node1_name = param.pretrained_model_node1
      node2_name = param.pretrained_model_node2
      if node1_name == '':
        node1_name = self.proto.node1
      if node2_name == '':
        node2_name = self.proto.node2
      mat = None
      for pretrained_model in param.pretrained_model:
        model = util.ReadModel(pretrained_model)
        edge = next(e for e in model.edge if e.node1 == node1_name and e.node2 == node2_name)
        pretrained_param = next(p for p in edge.param if p.name == param.name)
        assert pretrained_param.mat != '',\
                'Pretrained param %s in edge %s:%s of model %s is empty!!' % (
                  pretrained_param.name, edge.node1, edge.node2, pretrained_model)
        if param.transpose_pretrained:
          assert param.dimensions == pretrained_param.dimensions[::-1],\
              'Param has shape %s but transposed pretrained param has shape %s' % (
                param.dimensions, reversed(pretrained_param.dimensions))
        else:
          assert param.dimensions == pretrained_param.dimensions,\
              'Param has shape %s but pretrained param has shape %s' % (
                param.dimensions, pretrained_param.dimensions)
        this_mat = util.ParameterAsNumpy(pretrained_param)
        if param.transpose_pretrained:
          this_mat = this_mat.T
        if mat is None:
          mat = this_mat
        else:
          mat += this_mat
      return mat / len(param.pretrained_model)
    else:
      raise Exception('Unknown parameter initialization.')

  def SaveParameters(self):
    for param in self.proto.param:
      param.mat = util.NumpyAsParameter(self.params[param.name].asarray())

  def ConvOuter(self, grad):
    w = self.params['weight']
    conv = self.conv_params
    size = conv.size
    stride = conv.stride
    padding = conv.padding
    num_filters = conv.num_filters
    num_colors = conv.num_colors

    f, numdims = w.shape
    assert f == num_filters, 'f is %d but num_filters is %d' % (f, num_filters)
    assert numdims == size**2 * num_colors

    input_t = self.input_t
    if conv.max_pool:
      output_t = self.unpooled_layer
    else:
      output_t = self.output_t
    numimages, numdims = input_t.shape

    assert numdims % num_colors == 0
    x = int(np.sqrt(numdims / num_colors))
    assert x**2 == numdims/num_colors

    n_locs = (x + 2 * padding - size) / stride + 1

    self.node1.state.transpose(input_t)
    cc.convOutp(input_t, output_t, grad, n_locs, size, stride, num_colors)

  def UpdateParams(self, deriv, step):
    """ Update the parameters associated with this edge.

    Update the weights and associated parameters.
    Args:
      deriv: Gradient w.r.t the inputs at the outgoing end.
      step: Training step.
    """

    logging.debug('UpdateParams in edge %s', self.name)
    h = self.hyperparams
    numcases = self.node1.batchsize
    if h.momentum_change_steps > step:
      f = float(step) / h.momentum_change_steps
      momentum = (1.0 - f) * h.initial_momentum + f * h.final_momentum
    else:
      momentum = h.final_momentum
    epsilon = h.base_epsilon
    if h.epsilon_decay == deepnet_pb2.Hyperparams.INVERSE_T:
      epsilon = h.base_epsilon / (1 + float(step) / h.epsilon_decay_half_life)
    if step < h.start_learning_after:
      epsilon = 0.0

    sign = -1
    w_delta = self.params['grad_weight']
    w = self.params['weight']
    if h.adapt == deepnet_pb2.Hyperparams.NONE:
      w_delta.mult(momentum)
      if h.apply_l2_decay:
        w_delta.add_mult(w, (1-momentum) * h.l2_decay)
      if self.conv:
        self.ConvOuter(self.temp)
        w_delta.add_mult(self.temp, (1.0 - momentum) / numcases)
      else:
        w_delta.add_dot(self.node1.state, deriv.T, (1.0 - momentum) / numcases)
      w.add_mult(w_delta, -epsilon)
    else:
      raise Exception('Not implemented.')
    if h.apply_weight_norm:
      w.norm_limit(h.weight_norm, axis=0)

  def LoadParams(self, proto, node1, node2):
    """Load the parameters for this edge.

    Load the parameters if present in the proto. Otherwise initialize them
    appropriately.
    Args:
      proto: Protocol buffer object that describes this edge.
      node1: Layer object at the input to this edge.
      node2: Layer object at the output of this edge.
   """
    self.hyperparams = proto.hyperparams
    param_names = [param.name for param in proto.param]
    for param in proto.param:
      if 'grad_'+param.name not in param_names and not param.name.startswith(
        'grad_') and not param.name.startswith('gradstats_'):
        grad_p = deepnet_pb2.Parameter()
        grad_p.CopyFrom(param)
        grad_p.name = 'grad_' + param.name
        proto.param.extend([grad_p])
      if 'gradstats_'+param.name not in param_names and not param.name.startswith(
        'gradstats_') and not param.name.startswith('grad_'):
        grad_stats_p = deepnet_pb2.Parameter()
        grad_p.CopyFrom(param)
        grad_stats_p.name = 'gradstats_' + param.name
        grad_stats_p.constant = 1.0
        proto.param.extend([grad_stats_p])

    for param in proto.param:
      if not param.dimensions:
        if param.conv:
          self.conv = True
          self.conv_params = param.conv_params
          param.dimensions.extend(
            [self.conv_params.num_filters,
             self.conv_params.size**2 * self.conv_params.num_colors])
          self.input_t = cm.CUDAMatrix(np.zeros(node1.state.shape[::-1]))
          self.output_t = cm.CUDAMatrix(np.zeros(node2.state.shape[::-1]))
          self.output_t2 = cm.CUDAMatrix(np.zeros(node2.state.shape[::-1]))
          if param.conv_params.max_pool:
            numdims, numimages = node1.state.shape
            num_colors = param.conv_params.num_colors
            num_filters = param.conv_params.num_filters
            padding = param.conv_params.padding
            size = param.conv_params.size
            stride = param.conv_params.stride
            assert numdims % num_colors == 0
            x = int(np.sqrt(numdims / num_colors))
            assert x**2 == numdims / num_colors
            n_locs = (x + 2 * padding - size) / stride + 1
            self.unpooled_layer = cm.CUDAMatrix(np.zeros((numimages, n_locs**2 * num_filters)))
        else:
          param.dimensions.extend([node1.numlabels * node1.dimensions,
                                   node2.numlabels * node2.dimensions])
      if param.mat and 'grad' not in param.name:
        print 'Loading saved parameters'
        mat = util.ParameterAsNumpy(param)
      else:
        mat = self.InitializeParameter(param)
      self.params[param.name] = cm.CUDAMatrix(mat)
      if param.name == 'weight':
        self.temp = cm.empty(mat.shape)
        self.temp2 = cm.empty(mat.shape)
