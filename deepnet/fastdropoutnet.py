"""Implements a feed-forward neural net."""
from neuralnet import *
from layer import *

class FastDropoutNet(NeuralNet):

  def __init__(self, *args, **kwargs):
    super(FastDropoutNet, self).__init__(*args, **kwargs)
    self.SetUpLinks()

  def SetUpLinks(self):
    """Modifies self.net to create two parallel nets."""
    net1 = self.net
    net2 = CopyModel(net1)  # creates a copy of the net.
    for layer in net1.layer:
      if layer.hyperparams.fast_dropout:
        layer.is_output = True
      layer.name += '_1'
    for edge in net1.edge:
      edge.node1 += '_1'
      edge.node2 += '_1'
    for layer in net2.layer:
      layer.tied = True
      layer.tied_to = '%s_1' % layer.name
      if layer.hyperparams.fast_dropout:
        layer.is_output = True
      if layer.is_input or layer.is_output:
        layer.data_field.tied = True
        layer.data_field.tied_to = '%s_1' % layer.name
      layer.name += '_2'
    for edge in net2.edge:
      edge.tied = True
      edge.tied_to_node1 = '%s_1' % edge.node1
      edge.tied_to_node2 = '%s_1' % edge.node2
      edge.node1 += '_2'
      edge.node2 += '_2'
    self.net.MergeFrom(net2)

  def LoadModelOnGPU(self, *args, **kwargs):
    super(FastDropoutNet, self).LoadModelOnGPU(*args, **kwargs)
    for layer in self.layer:
      if layer.hyperparams.fast_dropout and layer.proto.tied:
        tied_to = next(l for l in self.layer if l.name == layer.proto.tied_to)
        layer.fast_dropout_partner = tied_to
        tied_to.fast_dropout_partner = layer

  def ComputeUp(self, layer, train=False, step=0, maxsteps=0):
    """
    Computes the state of `layer', given the state of its incoming neighbours.

    Args:
      layer: Layer whose state is to be computed.
      train: True if this computation is happening during training, False during
        evaluation.
      step: Training step.
      maxsteps: Maximum number of steps that will be taken (Needed because some
        hyperparameters may depend on this).
    """
    layer.dirty = False
    perf = None
    if layer.is_input or layer.is_initialized:
      layer.GetData()
    else:
      for i, edge in enumerate(layer.incoming_edge):
        if edge in layer.outgoing_edge:
          continue
        inputs = layer.incoming_neighbour[i].state
        if edge.conv or edge.local:
          if i == 0:
            ConvolveUp(inputs, edge, layer.state)
          else:
            AddConvoleUp(inputs, edge, layer.state)
        else:
          w = edge.params['weight']
          factor = edge.proto.up_factor
          if i == 0:
            cm.dot(w.T, inputs, target=layer.state)
            if factor != 1:
              layer.state.mult(factor)
          else:
            layer.state.add_dot(w.T, inputs, mult=factor)
      b = layer.params['bias']
      if layer.replicated_neighbour is None:
        layer.state.add_col_vec(b)
      else:
        layer.state.add_dot(b, layer.replicated_neighbour.NN)
      layer.ApplyActivation()
      if layer.hyperparams.sparsity:
        layer.state.sum(axis=1, target=layer.dimsize)
        perf = deepnet_pb2.Metrics()
        perf.MergeFrom(layer.proto.performance_stats)
        perf.count = layer.batchsize
        layer.dimsize.sum(axis=0, target=layer.unitcell)
        perf.sparsity = layer.unitcell.euclid_norm() / layer.dimsize.shape[0]
        layer.unitcell.greater_than(0)
        if layer.unitcell.euclid_norm() == 0:
          perf.sparsity *= -1

    if layer.hyperparams.fast_dropout:
      layer.data.assign(layer.state)

    if layer.hyperparams.dropout:
      if train and maxsteps - step >= layer.hyperparams.stop_dropout_for_last:
        # Randomly set states to zero.
        if layer.hyperparams.mult_dropout:
          layer.mask.fill_with_randn()
          layer.mask.add(1)
          layer.state.mult(layer.mask)
        else:
          layer.mask.fill_with_rand()
          layer.mask.greater_than(layer.hyperparams.dropout_prob)
          if layer.hyperparams.blocksize > 1:
            layer.mask.blockify(layer.hyperparams.blocksize)
          layer.state.mult(layer.mask)
      else:
        # Produce expected output.
        if layer.hyperparams.mult_dropout:
          pass
        else:
          layer.state.mult(1.0 - layer.hyperparams.dropout_prob)
    return perf

  def EvaluateOneBatch(self):
    """Evaluate one mini-batch."""
    losses = self.ForwardPropagate()
    losses.extend([node.GetLoss() for node in self.output_datalayer if '_1' in node.name])
    return losses

  def GetFastDropoutGradient(self, layer):
    perf = deepnet_pb2.Metrics()
    perf.MergeFrom(layer.proto.performance_stats)
    perf.count = layer.batchsize
    if layer.loss_function == deepnet_pb2.Layer.SQUARED_LOSS:
      target = layer.statesize
      layer.data.subtract(layer.fast_dropout_partner.data, target=target)
      error = target.euclid_norm()**2
      perf.error = error
      layer.deriv.add_mult(target, alpha=layer.loss_weight)
      layer.ComputeDeriv()
    else:
      raise Exception('Unknown loss function for ReLU units.')
    return perf

  def ComputeDown(self, layer, step):
    """Backpropagate through this layer.
    Args:
      step: The training step. Needed because some hyperparameters depend on
      which training step they are being used in.
    """
    if layer.is_input:  # Nobody to backprop to.
      return
    # At this point layer.deriv contains the derivative with respect to the
    # outputs of this layer. Compute derivative with respect to the inputs.
    h = layer.hyperparams
    loss = None
    if h.fast_dropout:
      if layer.hyperparams.sparsity:
        layer.AddSparsityGradient()
      loss = self.GetFastDropoutGradient(layer)
    else:
      if layer.is_output:
        loss = layer.GetLoss(get_deriv=True)
      else:
        if layer.hyperparams.sparsity:
          layer.AddSparsityGradient()
        layer.ComputeDeriv()
    # Now layer.deriv contains the derivative w.r.t to the inputs.
    # Send it down each incoming edge and update parameters on the edge.
    for edge in layer.incoming_edge:
      if edge.conv or edge.local:
        AccumulateConvDeriv(edge.node1, edge, layer.deriv)
      else:
        self.AccumulateDeriv(edge.node1, edge, layer.deriv)
      self.UpdateEdgeParams(edge, layer.deriv, step)
    # Update the parameters on this layer (i.e., the bias).
    self.UpdateLayerParams(layer, step)
    return loss

  def SetUpData(self, skip_outputs=False, skip_layernames=[]):
    """Setup the data."""
    hyp_list = []
    name_list = [[], [], []]
    for node in self.layer:
      if not (node.is_input or node.is_output):
        continue
      if skip_outputs and node.is_output:
        continue
      if node.name in skip_layernames:
        continue
      data_field = node.proto.data_field
      if node.hyperparams.fast_dropout:
        pass
        #self.fast_dropout_layers.append(node)
      elif data_field.tied:
        self.tied_datalayer.append(node)
        node.tied_to = next(l for l in self.datalayer\
                            if l.name == data_field.tied_to)
      else:
        self.datalayer.append(node)
        hyp_list.append(node.hyperparams)
        if data_field.train:
          name_list[0].append(data_field.train)
        if data_field.validation:
          name_list[1].append(data_field.validation)
        if data_field.test:
          name_list[2].append(data_field.test)
    if self.t_op:
      op = self.t_op
    else:
      op = self.e_op
    handles = GetDataHandles(op, name_list, hyp_list,
                             verbose=self.verbose)
    self.train_data_handler = handles[0]
    self.validation_data_handler = handles[1]
    self.test_data_handler = handles[2]


