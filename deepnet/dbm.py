"""Implements a Deep Boltzmann Machine."""
from neuralnet import *

class DBM(NeuralNet):

  def __init__(self, *args, **kwargs):
    super(DBM, self).__init__(*args, **kwargs)
    self.initializer_net = None
    self.cd = self.t_op.optimizer == deepnet_pb2.Operation.CD

  @staticmethod
  def AreInputs(l):
    return reduce(lambda a, x: x.is_input and a, l, True)

  def SetPhase(self, layer, pos=True):
    """Setup required before starting a phase.

    This method makes 'state' and 'sample' point to the right variable depending
    on the phase.
    """
    if pos:
      layer.state = layer.pos_state
      layer.sample = layer.pos_sample
    else:
      layer.state = layer.neg_state
      layer.sample = layer.neg_sample

  def DumpModelState(self, step):
    state_dict = dict([(node.name, node.state.asarray().T) for node in self.node_list])
    filename = '/ais/gobi3/u/nitish/flickr/states/%s_%d' % (self.net.name, step)
    print 'Dumping state at step %d to %s' % (step, filename)
    np.savez(filename, **state_dict)

  def Sort(self):
    """Sort layers into useful orders.

    After this method is done:
    pos_phase_order: Order in which nodes have to be updated in the positive
      phase.
    neg_phase_order: Order in which nodes have to be updated in the negative
      phase.
    node_list: List of all nodes. All input nodes occur before non input ones.
    """
    non_input_nodes = []
    node_list = list(self.input_datalayer)
    for node in self.layer:
      if not node.is_input:
        non_input_nodes.append(node)
    node_list.extend(non_input_nodes)
    if self.net.positive_phase_order:
      self.pos_phase_order = [self.GetLayerByName(x) for x in self.net.positive_phase_order]
      self.pos_phase_order.extend([self.GetLayerByName(x) for x in self.unclamped_layer])
    else:
      self.pos_phase_order = non_input_nodes
    if self.net.negative_phase_order:
      self.neg_phase_order = [self.GetLayerByName(x) for x in self.net.negative_phase_order]
    else:
      self.neg_phase_order = node_list
    return node_list

  def ComputeUnnormalizedLogProb(self):
    pass

  def ComputeUp(self, layer, train=False, compute_input=False, step=0,
                maxsteps=0, use_samples=False, neg_phase=False):
    """
    Computes the state of a layer, given the state of its incoming neighbours.

    Args:
      train: True if this computation is happening during training, False during
        evaluation.
      compute_input: If True, the state of the input layer will be computed.
        Otherwise, it will be loaded as data.
      step: Training step.
      maxsteps: Maximum number of steps that will be taken (Some hyperparameters
        may depend on this.)
      use_samples: Use neighbours' samples to update the layer's state.
  """
    if layer.is_input and not compute_input:
      layer.GetData()
    else:
      for i, edge in enumerate(layer.incoming_edge):
        neighbour = layer.incoming_neighbour[i]
        if use_samples:
          inputs = neighbour.sample
        else:
          inputs = neighbour.state
        if edge.node2 == layer:
          w = edge.params['weight'].T
          factor = edge.proto.up_factor
        else:
          w = edge.params['weight']
          factor = edge.proto.down_factor
        if i == 0:
          cm.dot(w, inputs, target=layer.state)
          if factor != 1:
            layer.state.mult(factor)
        else:
          layer.state.add_dot(w, inputs, mult=factor)
      b = layer.params['bias']
      if layer.replicated_neighbour is None:
        layer.state.add_col_vec(b)
      else:
        layer.state.add_dot(b, layer.replicated_neighbour.NN)
      layer.ApplyActivation()
    if layer.hyperparams.dropout:
      if train and maxsteps - step >= layer.hyperparams.stop_dropout_for_last:
        # Randomly set states to zero.
        if not neg_phase:
          layer.mask.fill_with_rand()
          layer.mask.greater_than(layer.hyperparams.dropout_prob)
        layer.state.mult(layer.mask)
      else:
        # Produce expected output.
        layer.state.mult(1.0 - layer.hyperparams.dropout_prob)


  def PositivePhase(self, train=False, evaluate=False, step=0):
    """Perform the positive phase.

    This method computes the sufficient statistics under the data distribution.
    """

    # Do a forward pass in the initializer net, if set.
    if self.initializer_net:
      self.initializer_net.ForwardPropagate(train=train, step=step)

    # Initialize layers.
    for node in self.node_list:
      if node.is_input:
        # Load data into input nodes.
        self.ComputeUp(node, train=train)
      elif node.is_initialized:
        node.state.assign(node.initialization_source.state)
      else:
        # Initialize other nodes to zero.
        node.ResetState(rand=False)

    # Starting MF.
    for i in range(self.net.hyperparams.mf_steps):
      for node in self.pos_phase_order:
        self.ComputeUp(node, train=train, step=step, maxsteps=self.train_stop_steps)
    # End of MF.

    losses = []
    if train:
      for node in self.layer:
        r = node.CollectSufficientStatistics()
        if r is not None:  # This is true only if sparsity is active.
          perf = deepnet_pb2.Metrics()
          perf.MergeFrom(node.proto.performance_stats)
          perf.count = 1
          perf.sparsity = r
          losses.append(perf)
      for edge in self.edge:
        edge.CollectSufficientStatistics()

    # Evaluation
    # If CD, then this step would be performed by the negative phase anyways,
    # So the loss is measured in the negative phase instead. Return []
    # Otherwise, reconstruct the input given the other layers and report
    # the loss.
    if not self.cd or evaluate:
      for node in self.input_datalayer:
        self.ComputeUp(node, compute_input=True, step=step, maxsteps=self.train_stop_steps)
        losses.append(node.GetLoss())
    return losses

  def InitializeNegPhase(self, to_pos=False):
    """Initialize negative particles.

    Copies the pos state and samples it to initialize the ngative particles.
    """
    for layer in self.layer:
      self.SetPhase(layer, pos=False)
      if to_pos:
        layer.state.assign(layer.pos_state)
      else:
        layer.ResetState(rand=True)
      layer.Sample()
      self.SetPhase(layer, pos=True)

  def NegativePhase(self, step=0, train=True, gibbs_steps=-1):
    """Perform the negative phase.

    This method computes the sufficient statistics under the model distribution.
    Args:
      step: Training step
      train: If true, then this computation is happening during training.
      gibbs_steps: Number of gibbs steps to take. If -1, use default.
    """
    losses = []

    if self.cd:
      for node in self.node_list:
        if not node.is_input:
          node.Sample()
    else:
      for node in self.layer:
        self.SetPhase(node, pos=False)

    if gibbs_steps < 0:
      h = self.net.hyperparams
      start_after = h.start_step_up_cd_after
      if start_after > 0 and start_after < step:
        gibbs_steps = h.gibbs_steps + 1 + (step - start_after) / h.step_up_cd_after
      else:
        gibbs_steps = h.gibbs_steps

    for i in range(gibbs_steps):
      for node in self.neg_phase_order:
        self.ComputeUp(node, train=train, step=step,
                       maxsteps=self.train_stop_steps, use_samples=True,
                       compute_input=True, neg_phase=True)
        if i == 0 and node.is_input and self.cd:
          losses.append(node.GetLoss())
        if node.is_input:
          if node.sample_input and node.hyperparams.sample_input_after <= step:
            node.Sample()
          else:
            # Not sampling inputs usually makes learning faster.
            node.sample.assign(node.state)
        else:
          node.Sample()
    # End of Gibbs Sampling.

    if train:
      for node in self.layer:
        node.CollectSufficientStatistics(neg=True)
        self.UpdateLayerParams(node, step=step)
      for edge in self.edge:
        edge.CollectSufficientStatistics(neg=True)
        self.UpdateEdgeParams(edge, step=step)

    if not self.cd:
      for node in self.layer:
        self.SetPhase(node, pos=True)
    return losses

  def UpdateLayerParams(self, layer, step=0):
    """Update parameters associated with this layer."""
    layer.gradient.add_mult(layer.suff_stats, -1.0 / layer.batchsize)
    if layer.tied_to:
      layer.tied_to.gradient.add(layer.gradient)
      layer.gradient.assign(0)
      layer = layer.tied_to
    layer.num_grads_received += 1
    if layer.num_grads_received == layer.num_shares:
      layer.Update('bias', step, no_reg=True)  # By default, do not regularize bias.

  def UpdateEdgeParams(self, edge, step):
    """ Update the parameters associated with this edge."""
    numcases = edge.node1.batchsize
    edge.gradient.add_mult(edge.suff_stats, -1.0/numcases)
    if edge.tied_to:
      edge.tied_to.gradient.add(edge.gradient)
      edge.gradient.assign(0)
      edge = edge.tied_to
    edge.num_grads_received += 1
    if edge.num_grads_received == edge.num_shares:
      edge.Update('weight', step)

  def GetBatch(self, handler=None):
    super(DBM, self).GetBatch(handler=handler)
    if self.initializer_net:
      self.initializer_net.GetBatch()

  def TrainOneBatch(self, step):
    losses1 = self.PositivePhase(train=True, step=step)
    if step == 0 and self.t_op.optimizer == deepnet_pb2.Operation.PCD:
      self.InitializeNegPhase(to_pos=True)
    losses2 = self.NegativePhase(step, train=True)
    losses1.extend(losses2)
    return losses1
  
  def EvaluateOneBatch(self):
    losses = self.PositivePhase(train=False, evaluate=True)
    return losses

  def SetUpData(self, *args, **kwargs):
    super(DBM, self).SetUpData(*args, **kwargs)

    # Set up data for initializer net.
    if self.initializer_net:
      for node in self.initializer_net.layer:
        try:
          matching_dbm_node = next(l for l in self.layer \
                              if l.name == node.name)
        except StopIteration:
          matching_dbm_node = None
        if matching_dbm_node:
          if node.is_input or node.is_output:
            self.initializer_net.tied_datalayer.append(node)
            node.tied_to = matching_dbm_node
          elif matching_dbm_node.is_initialized:
            matching_dbm_node.initialization_source = node


  def LoadModelOnGPU(self, batchsize=-1):
    super(DBM, self).LoadModelOnGPU(batchsize=batchsize)
    if self.net.initializer_net:
      self.initializer_net = NeuralNet(self.net.initializer_net, self.t_op,
                                     self.e_op)
      self.initializer_net.LoadModelOnGPU(batchsize=batchsize)

  def Reconstruct(self, layername, numbatches, inputlayername=[],
                  validation=True):
    """Reconstruct from the model.
    Args:
      layername: Name of the layer which is to be reconstructed.
      numbatches: Number of batches to reconstruct.
      inputlayername: List of input layers whose states will be returned.
      validation: If True, reconstruct the validation set,
        else reconstruct test set.
    Returns:
      The reconstruction for layer 'layername' and inputs in layers
        'inputlayername'
    """
    step = 0
    self.recon = []
    self.inputs = []
    self.recon_pos = 0
    inputlayer = []
    layer_to_tap = self.GetLayerByName(layername, down=True)
    self.recon = np.zeros((numbatches * self.e_op.batchsize,
                           layer_to_tap.state.shape[0]))
    for i, lname in enumerate(inputlayername):
      l = self.GetLayerByName(lname)
      inputlayer.append(l)
      self.inputs.append(np.zeros((numbatches * self.e_op.batchsize,
                                   l.state.shape[0])))
    if validation:
      datagetter = self.GetValidationBatch
    else:
      datagetter = self.GetTestBatch
    for batch in range(numbatches):
      datagetter()
      self.ReconstructOneBatch(layer_to_tap, inputlayer)
    return self.recon, self.inputs

  def GetAllRepresentations(self, numbatches, validation=True):
    """Get representations at all layers.
    Returns:
      A dictionary with the name of the layer as the key and its state as as the
        value.
    """
    if validation:
      datagetter = self.GetValidationBatch
    else:
      datagetter = self.GetTestBatch
    rep_list = []
    names = []
    for node in self.node_list:
      rep_list.append(np.zeros((numbatches * node.state.shape[1],
                                node.state.shape[0]), dtype='float32'))
      names.append(node.name)
    for batch in range(numbatches):
      datagetter()
      self.PositivePhase(train=False, evaluate=False)
      for i, node in enumerate(self.node_list):
        rep_list[i][batch*node.batchsize:(batch+1)*node.batchsize,:] =\
            node.state.asarray().T
    return dict(zip(names, rep_list))

  def WriteRepresentationToDisk(self, layernames, output_dir, memory='1G',
                                dataset='test', input_recon=False):
    layers = [self.GetLayerByName(lname) for lname in layernames]
    numdim_list = [layer.state.shape[0] for layer in layers]
    if dataset == 'train':
      datagetter = self.GetTrainBatch
      if self.train_data_handler is None:
        return
      numbatches = self.train_data_handler.num_batches
      size = numbatches * self.train_data_handler.batchsize
    elif dataset == 'validation':
      datagetter = self.GetValidationBatch
      if self.validation_data_handler is None:
        return
      numbatches = self.validation_data_handler.num_batches
      size = numbatches * self.validation_data_handler.batchsize
    elif dataset == 'test':
      datagetter = self.GetTestBatch
      if self.test_data_handler is None:
        return
      numbatches = self.test_data_handler.num_batches
      size = numbatches * self.test_data_handler.batchsize
    datawriter = DataWriter(layernames, output_dir, memory, numdim_list, size)

    for batch in range(numbatches):
      datagetter()
      sys.stdout.write('\r%d' % (batch+1))
      sys.stdout.flush()
      self.PositivePhase(train=False, evaluate=input_recon)
      reprs = [l.state.asarray().T for l in layers]
      datawriter.Submit(reprs)
    sys.stdout.write('\n')
    size = datawriter.Commit()
    return size

  def GetRepresentation(self, layername, numbatches, inputlayername=[],
                        validation=True):
    """Get the representation at layer 'layername'."""
    step = 0
    self.rep_pos = 0
    inputlayer = []
    self.inputs = []
    layer_to_tap = self.GetLayerByName(layername)
    self.rep = np.zeros((numbatches * self.e_op.batchsize, layer_to_tap.state.shape[0]))
    for i, lname in enumerate(inputlayername):
      l = self.GetLayerByName(lname)
      inputlayer.append(l)
      self.inputs.append(np.zeros((numbatches * self.e_op.batchsize,
                                   l.state.shape[0])))
    if validation:
      datagetter = self.GetValidationBatch
    else:
      datagetter = self.GetTestBatch
    for batch in range(numbatches):
      datagetter()
      self.GetRepresentationOneBatch(layer_to_tap, inputlayer)
    return self.rep, self.inputs

  def GetLayerByName(self, layername, down=False):
    try:
      l = next(l for l in self.layer if l.name == layername)
    except StopIteration:
      l = None
    return l

  def Inference(self, steps, layernames, unclamped_layers, output_dir, memory='1G', dataset='test', method='gibbs'):
    layers_to_infer = [self.GetLayerByName(l) for l in layernames]
    layers_to_unclamp = [self.GetLayerByName(l) for l in unclamped_layers]

    numdim_list = [layer.state.shape[0] for layer in layers_to_infer]
    for l in layers_to_unclamp:
      l.is_input = False
      self.pos_phase_order.append(l)

    if dataset == 'train':
      datagetter = self.GetTrainBatch
      if self.train_data_handler is None:
        return
      numbatches = self.train_data_handler.num_batches
      size = numbatches * self.train_data_handler.batchsize
    elif dataset == 'validation':
      datagetter = self.GetValidationBatch
      if self.validation_data_handler is None:
        return
      numbatches = self.validation_data_handler.num_batches
      size = numbatches * self.validation_data_handler.batchsize
    elif dataset == 'test':
      datagetter = self.GetTestBatch
      if self.test_data_handler is None:
        return
      numbatches = self.test_data_handler.num_batches
      size = numbatches * self.test_data_handler.batchsize
    dw = DataWriter(layernames, output_dir, memory, numdim_list, size)

    gibbs = method == 'gibbs'
    mf = method == 'mf'

    for batch in range(numbatches):
      sys.stdout.write('\r%d' % (batch+1))
      sys.stdout.flush()
      datagetter()
      for node in self.node_list:
        if node.is_input or node.is_initialized:
          node.GetData()
        else:
          node.ResetState(rand=False)
        if gibbs:
          node.sample.assign(node.state)
      for i in range(steps):
        for node in self.pos_phase_order:
          self.ComputeUp(node, use_samples=gibbs)
          if gibbs:
            node.Sample()
      output = [l.state.asarray().T for l in layers_to_infer]
      dw.Submit(output)
    sys.stdout.write('\n')
    size = dw.Commit()
    return size[0]

  def ReconstructOneBatch(self, layer, inputlayers):
    self.PositivePhase(train=False, evaluate=True)
    self.recon[self.recon_pos:self.recon_pos + self.e_op.batchsize,:] =\
      layer.state.asarray().T
    for i, l in enumerate(inputlayers):
      self.inputs[i][self.recon_pos:self.recon_pos + self.e_op.batchsize,:] =\
        l.data.asarray().T
    self.recon_pos += self.e_op.batchsize

  def GetRepresentationOneBatch(self, layer, inputlayers):
    self.PositivePhase(train=False, evaluate=False)
    if layer.proto.is_input:
      self.rep[self.rep_pos:self.rep_pos + self.e_op.batchsize,:] =\
          layer.data.asarray().T
    else:
      self.rep[self.rep_pos:self.rep_pos + self.e_op.batchsize,:] =\
          layer.state.asarray().T
    for i, l in enumerate(inputlayers):
      self.inputs[i][self.rep_pos:self.rep_pos + self.e_op.batchsize,:] =\
        l.data.asarray().T
    self.rep_pos += self.e_op.batchsize

  def UnclampLayer(self, layername):
    """Unclamps the layer 'layername'.
    
    Most useful when called just after calling the constructor.
    """
    for l in self.net.layer:
      if l.name == layername:
        print 'Unclamping %s' % layername
        l.is_input = False
        self.unclamped_layer.append(l.name)
