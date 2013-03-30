"""Implements an edge connecting two layers of neurons."""
import cudamat as cm
from cudamat_conv import cudamat_conv2 as cc
import numpy as np
import deepnet_pb2
import visualize
import logging
import util
import pdb
import os

class Edge(object):
  def __init__(self, proto, node1, node2, t_op=None, tied_to=None):
    self.node1 = node1
    self.node2 = node2
    self.transpose = False
    self.tied_to = tied_to
    if proto.tied:
      tied_to.num_shares += 1
      self.transpose = proto.tied_transpose
      proto.CopyFrom(tied_to.proto)
      proto.node1 = node1.name
      proto.node2 = node2.name
      for param in proto.param:
        if param.dimensions:
          dims = list(reversed(param.dimensions))
          del param.dimensions[:]
          param.dimensions.extend(dims)
    self.proto = proto
    self.params = {}
    self.conv = False
    self.local = False
    self.t_op = t_op
    self.name = '%s:%s' % (self.node1.name, self.node2.name)
    self.prefix = proto.prefix
    self.hyperparams = None
    if proto.directed:
      node1.AddOutgoingEdge(self)
      node2.AddIncomingEdge(self)
    else:
      node1.AddOutgoingEdge(self)
      node1.AddIncomingEdge(self)
      node2.AddOutgoingEdge(self)
      node2.AddIncomingEdge(self)

    self.LoadParams(tied_to=tied_to)
    self.marker = 0
    self.fig = visualize.GetFigId()
    self.fig_stats = visualize.GetFigId()
    if self.conv or self.local:
      self.conv_filter_fig = visualize.GetFigId()
    self.num_shares = 1
    self.num_grads_received = 0

  def Show(self):
    if not self.hyperparams.enable_display:
      return
    if self.node1.is_input:
      if self.conv or self.local:
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
          visualize.show_stats(self, self.fig_stats, self.name)
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
      if param.conv or param.local:
        fan_in = np.prod(param.dimensions[0])
      else:
        fan_in = np.prod(param.dimensions[1])
      stddev = param.sigma / np.sqrt(fan_in)
      return stddev * np.random.randn(*tuple(param.dimensions))
    elif param.initialization == deepnet_pb2.Parameter.DENSE_UNIFORM_SQRT_FAN_IN:
      if param.conv or param.local:
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

      if param.transpose_pretrained:
        temp = node1_name
        node1_name = node2_name
        node2_name = temp
      mat = None
      for pretrained_model in param.pretrained_model:
        model_file = os.path.join(self.prefix, pretrained_model)
        model = util.ReadModel(model_file)
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

  def AllocateBatchsizeDependentMemory(self):
    for param in self.proto.param:
      if param.conv or param.local:
        self.AllocateMemoryForConvolutions(param, self.node1, self.node2)
 
  def AllocateMemoryForConvolutions(self, param, node1, node2):
    self.conv = param.conv
    self.local = param.local
    if self.conv:
      assert not self.local
    else:
      assert not self.conv
    self.conv_params = param.conv_params
    num_colors = self.conv_params.num_colors
    num_filters = self.conv_params.num_filters
    size = self.conv_params.size
    padding = self.conv_params.padding
    stride = self.conv_params.stride

    numdims, numimages = node1.state.shape
    assert numdims % num_colors == 0
    x = int(np.sqrt(numdims / num_colors))
    assert x**2 == numdims / num_colors
    n_locs = (x + 2 * padding - size) / stride + 1

    input_shape = node1.state.shape[::-1]
    output_shape = node2.state.shape[::-1]

    self.input_t = cm.empty(input_shape)
    self.input_t2 = cm.empty(input_shape)
    self.output_t = cm.empty(output_shape)
    self.output_t2 = cm.empty(output_shape)
    if param.conv_params.max_pool:
      pool_output_size = n_locs**2 * num_filters
      self.unpooled_layer = cm.empty((numimages, pool_output_size))
      pool_size = param.conv_params.pool_size
      pool_stride = param.conv_params.pool_stride
      n_pool_locs = (n_locs - pool_size) / pool_stride + 1
      assert output_shape[1] == n_pool_locs**2 * num_filters
      if param.conv_params.prob:
        self.rnd = cm.empty(self.unpooled_layer.shape)
    else:
      assert output_shape[1] == n_locs**2 * num_filters
    if param.conv_params.rnorm:
      self.unrnormalized_layer = cm.empty(output_shape)
      self.denoms = cm.empty(output_shape)
      self.rnorm_temp1 = cm.empty(output_shape)
      self.rnorm_temp2 = cm.empty(output_shape)

    return n_locs

  def LoadParams(self, tied_to=None):
    """Load the parameters for this edge.

    Load the parameters if present in self.proto. Otherwise initialize them
    appropriately.
    """
    node1 = self.node1
    node2 = self.node2
    proto = self.proto
    self.hyperparams = proto.hyperparams
    param_names = [param.name for param in proto.param]
    for param in proto.param:
      if param.conv or param.local:
        n_locs = self.AllocateMemoryForConvolutions(param, node1, node2)
      if not param.dimensions:
        if param.conv:
          cv = param.conv_params
          dims = [cv.num_filters, cv.size**2 * cv.num_colors]
        elif param.local:
          dims = [cv.num_filters, n_locs**2 * cv.size**2 * cv.num_colors]
        else:
          dims = [node1.numlabels * node1.dimensions,
                  node2.numlabels * node2.dimensions]
        param.dimensions.extend(dims)
      if tied_to:
        if self.transpose:
          self.params[param.name] = tied_to.params[param.name].T
        else:
          self.params[param.name] = tied_to.params[param.name]
        mat = self.params[param.name]
      else:
        if param.mat:  # and 'grad' not in param.name:
          mat = util.ParameterAsNumpy(param)
        else:
          mat = self.InitializeParameter(param)
        self.params[param.name] = cm.CUDAMatrix(mat)
      if param.name == 'weight':
        self.temp = cm.empty(mat.shape)
        #self.temp2 = cm.empty(mat.shape)
        self.gradient = cm.empty(mat.shape)
        self.grad_weight = cm.empty(mat.shape)
        self.gradient.assign(0)
        self.grad_weight.assign(0)
    if self.t_op and (self.t_op.optimizer == deepnet_pb2.Operation.PCD or \
      self.t_op.optimizer == deepnet_pb2.Operation.CD):
      self.suff_stats = cm.empty((self.node1.numlabels * self.node1.dimensions,
                                  self.node2.numlabels * self.node2.dimensions))

  def CollectSufficientStatistics(self, neg=False):
    logging.debug('Collecting suff stats %s', self.name)
    if self.node1.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.node1.state.div_by_row(self.node1.NN)
    if self.node2.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.node2.state.div_by_row(self.node2.NN)
    if not neg:
      h1 = self.node1.hyperparams
      h2 = self.node2.hyperparams
      if h1.sparsity:
        self.node1.state.add_col_mult(self.node1.means_temp, -1)
      if h2.sparsity:
        self.node2.state.add_col_mult(self.node2.means_temp, -1)
      cm.dot(self.node1.state, self.node2.state.T, target=self.suff_stats)
      if h1.sparsity:
        self.node1.state.add_col_vec(self.node1.means_temp)
      if h2.sparsity:
        self.node2.state.add_col_vec(self.node2.means_temp)
    else:
      self.suff_stats.add_dot(self.node1.state, self.node2.state.T, mult=-1.0)
    if self.node1.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.node1.state.mult_by_row(self.node1.NN)
    if self.node2.activation == deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
      self.node2.state.mult_by_row(self.node2.NN)

