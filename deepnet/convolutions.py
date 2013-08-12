"""Convolutional operations."""
from choose_matrix_library import *
import math

def ConvolveUp(inputs, edge, target):
  w = edge.params['weight']
  conv = edge.conv_params
  size = conv.size
  stride = conv.stride
  padding = conv.padding
  num_filters = conv.num_filters
  num_colors = conv.num_colors

  f, numdims = w.shape
  assert f == num_filters, 'f is %d but num_filters is %d' % (f, num_filters)
  if edge.conv:
    assert numdims == size**2 * num_colors

  input_t = edge.input_t
  inputs.transpose(input_t)
  # Convolve Up.
  if conv.max_pool:
    output_t = edge.unpooled_layer
  elif conv.rnorm:
    output_t = edge.unrnormalized_layer
  else:
    output_t = edge.output_t

  numimages, numdims = input_t.shape
  numimages2, numdims2 = output_t.shape
  assert numimages == numimages2, '%d %d.' % (numimages, numimages2)
  assert numdims % num_colors == 0
  x = int(math.sqrt(numdims / num_colors))
  assert x**2 == numdims/num_colors
  n_locs = (x + 2 * padding - size) / stride + 1
  if edge.conv:
    cc.convUp(input_t, w, output_t, n_locs, padding, stride, num_colors)
  else:
    cc.localUp(input_t, w, output_t, n_locs, padding, stride, num_colors)

  # Do maxpooling
  if conv.max_pool:
    input_t = output_t
    if conv.rnorm:
      output_t = edge.unrnormalized_layer
    else:
      output_t = edge.output_t
    n_locs = (n_locs - conv.pool_size) / conv.pool_stride + 1
    if conv.prob:
      rnd = edge.rnd
      rnd.fill_with_rand()
      cm.log(rnd)
      rnd.mult(-1)
      #cm.log(rnd)
      #rnd.mult(-1)
      cc.ProbMaxPool(input_t, rnd, output_t, num_filters, conv.pool_size, 0, conv.pool_stride, n_locs)
    else:
      cc.MaxPool(input_t, output_t, num_filters, conv.pool_size, 0, conv.pool_stride, n_locs)
  if conv.rnorm:
    input_t = output_t
    output_t = edge.output_t
    denoms = edge.denoms
    sizeX = conv.norm_size
    add_scale = conv.add_scale
    pow_scale = conv.pow_scale
    cc.ResponseNorm(input_t, denoms, output_t, num_filters, sizeX, add_scale, pow_scale)
  output_t.transpose(target)

def AccumulateConvDeriv(layer, edge, deriv):
  """Accumulate the derivative w.r.t the outputs of this layer.

  Each layer needs to compute derivatives w.r.t its outputs. These outputs may
  have been connected to lots of other nodes through outgoing edges.
  This method adds up the derivatives contributed by each outgoing edge.
  It gets derivatives w.r.t the inputs at the other end of an outgoing edge.
  Args:
    edge: The edge which is sending the derivative.
    deriv: The derivative w.r.t the inputs at the other end of this edge.
  """

  if layer.dirty:  # If some derivatives have already been received.
    raise Exception('Not implemented.')
  layer.dirty = True
  w = edge.params['weight']
  conv = edge.conv_params
  size = conv.size
  stride = conv.stride
  padding = conv.padding
  num_filters = conv.num_filters
  num_colors = conv.num_colors

  input_t = edge.input_t
  numImages, numdims = input_t.shape

  assert numdims % num_colors == 0
  x = int(math.sqrt(numdims / num_colors))
  assert x**2 == numdims/num_colors

  n_locs = (x + 2 * padding - size) / stride + 1

  # Incoming gradient.
  deriv.transpose(edge.output_t2)
  input_grads = edge.output_t2

  # Output activation (after conv + pool? + norm?)
  output_acts = edge.output_t

  if conv.rnorm:

    # ResponseNormUndo overwrites input_acts, so make a copy.
    input_acts = edge.rnorm_temp1
    input_acts.assign(edge.unrnormalized_layer)

    output_grads = edge.rnorm_temp2
    denoms = edge.denoms

    sizeX = conv.norm_size
    pow_scale = conv.pow_scale
    add_scale = conv.add_scale
    cc.ResponseNormUndo(input_grads, denoms, output_acts, input_acts,
                        output_grads, num_filters, sizeX, add_scale,
                        pow_scale)
    input_grads = output_grads
    output_acts = edge.unrnormalized_layer

  if conv.max_pool:
    input_acts = edge.unpooled_layer
    output_grads = edge.unpooled_layer
    # It's OK to overwrite input_acts because we don't need it later.

    n_pool_locs = (n_locs - conv.pool_size) / conv.pool_stride + 1
    sizeX = conv.pool_size
    strideX = conv.pool_stride
    cc.MaxPoolUndo(output_grads, input_acts, input_grads, output_acts, sizeX,
                   0, strideX, n_pool_locs)
    input_grads = output_grads
    output_acts = input_acts
  if layer.is_input:
    return

  output_grads = edge.input_t2
  if edge.conv:
    cc.convDown(input_grads, w, output_grads, n_locs, padding, stride, size, x, num_colors)
  else:
    cc.localDown(input_grads, w, output_grads, n_locs, padding, stride, size, x, num_colors)
  output_grads.transpose(layer.deriv)

def ConvOuter(edge, grad):
  """Get the gradient for the weights in this edge.
  Args:
    grad: (output) the gradient for the weights in this edge.
  """
  w = edge.params['weight']
  conv = edge.conv_params
  size = conv.size
  stride = conv.stride
  padding = conv.padding
  num_filters = conv.num_filters
  num_colors = conv.num_colors

  f, numdims = w.shape
  assert f == num_filters, 'f is %d but num_filters is %d' % (f, num_filters)
  if edge.conv:
    assert numdims == size**2 * num_colors
  input_t = edge.input_t
  if conv.max_pool:
    output_t = edge.unpooled_layer
  elif conv.rnorm:
    output_t = edge.rnorm_temp2
  else:
    output_t = edge.output_t2
  numdims, numimages = edge.node1.state.shape

  assert numdims % num_colors == 0
  x = int(math.sqrt(numdims / num_colors))

  assert x**2 == numdims/num_colors

  n_locs = (x + 2 * padding - size) / stride + 1

  if edge.conv:
    cc.convOutp(input_t, output_t, grad, n_locs, padding, size, stride, num_colors)
  else:
    cc.localOutp(input_t, output_t, grad, n_locs, padding, size, stride, num_colors)

def AddConvolveUp(inputs, edge, target):
  raise Exception('Not implemented.')

