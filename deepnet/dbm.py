"""Implements a Deep Boltzmann Machine."""
from neuralnet import *
from dbm_layer import *
from dbm_edge import *
import logging

class DBM(NeuralNet):

  @staticmethod
  def AreInputs(l):
    return reduce(lambda a, x: x.is_input and a, l, True)

  def SetLayerAndEdgeClass(self):
    self.LayerClass = DBMLayer
    self.EdgeClass = DBMEdge

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

  def PositivePhase(self, train=False, evaluate=False, step=0):
    """Perform the positive phase.

    This method computes the sufficient statistics under the data distribution.
    """
    logging.debug('Positive Phase')

    # Tell everyone that the positive phase is starting.
    for node in self.node_list:
      node.SetPhase(pos=True)

    # Starting MF.
    for node in self.node_list:
      if node.is_input:
        # Load data into input nodes.
        node.ComputeUp(train=train)
      elif node.is_initialized:
        node.GetData()
      else:
        # Initialize other nodes to zero.
        node.ResetState(rand=False)

    for i in range(self.net.hyperparams.mf_steps):
      for node in self.pos_phase_order:
        node.ComputeUp(train=train, step=step, maxsteps=self.train_stop_steps)
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
        edge.CollectSufficientStatistics(pos=True)

    # Evaluation
    # If CD, then this step would be performed by the negative phase anyways,
    # So the loss is measured in the negative phase instead. Return []
    # Otherwise, reconstruct the input given the other layers and report
    # the loss.
    if self.t_op.optimizer == deepnet_pb2.Operation.PCD or evaluate:
      for node in self.input_datalayer:
        node.ComputeUp(recon=True, step=step, maxsteps=self.train_stop_steps)
        losses.append(node.GetLoss())
    return losses

  def NegativePhase(self, step=0, train=True, gibbs_steps=-1):
    """Perform the negative phase.

    This method computes the sufficient statistics under the model distribution.
    Args:
      step: Training step
      train: If true, then this computation is happening during training.
      gibbs_steps: Number of gibbs steps to take. If -1, use default.
    """
    logging.debug('Negative Phase')
    losses = []
    # Tell everyone we are doing the negative phase.
    for node in self.node_list:
      node.SetPhase(pos=False)

    if self.t_op.optimizer == deepnet_pb2.Operation.CD:
      for node in self.node_list:
        if not node.is_input:
          node.Sample()

    step_up_after = self.net.hyperparams.step_up_cd_after
    if gibbs_steps < 0:
      if step_up_after > 0:
        gibbs_steps = self.net.hyperparams.gibbs_steps + step / step_up_after
      else:
        gibbs_steps = self.net.hyperparams.gibbs_steps

    for i in range(gibbs_steps):
      for node in self.neg_phase_order:
        node.ComputeUp(train=train, step=step, maxsteps=self.train_stop_steps)
        if (i == 0 and node.is_input
            and self.t_op.optimizer == deepnet_pb2.Operation.CD):
          losses.append(node.GetLoss())
        if node.is_input:
          # Not sampling inputs usually makes learning faster.
          node.sample.assign(node.state)
        else:
          node.Sample()
    # End of Gibbs Sampling.

    if train:
      for node in self.layer:
        node.CollectSufficientStatistics()
        node.UpdateParams(step=step)
      for edge in self.edge:
        edge.CollectSufficientStatistics(pos=False)
        edge.UpdateParams(step=step)

    # Reset phase.
    for node in self.node_list:
      node.SetPhase(pos=True)
    return losses

  def Infer(self, steps=0, method='mf', dumpstate=False):
    """Do Inference, conditioned on the inputs.

    This method performs MF / Gibbs sampling for the hidden units conditioned on the
    inputs.
    Args:
      steps: Number of steps of MF / Gibbs Sampling to perform. 0 means use
        the model's default value that was used during training.
      method: 'mf' or 'gibbs'.
      dumpstate: Dump the model's state to disk at each step.
    """
    logging.debug('Running Inference')
    if steps == 0:
      if method == 'mf':
        steps = self.net.hyperparams.mf_steps
      else:
        steps = self.net.hyperparams.gibbs_steps

    # Load data
    for node in self.node_list:
      if node.is_input:
        node.SetPhase(pos=True)
        node.ComputeUp()  # Loads data into pos_state.

    if method == 'mf':
      for node in self.node_list:
        node.SetPhase(pos=True)
        if not node.is_input:
          node.ResetState(rand=True)
    else:
      # Switch to negative phase and initialize samples.
      for node in self.node_list:
        node.SetPhase(pos=False)
        if node.is_input:
          node.sample.assign(node.pos_state)
          node.state.assign(node.pos_state)
        else:
          node.ResetState(rand=True)

    for i in range(steps):
      for node in self.pos_phase_order:
        node.ComputeUp()
        if method == 'gibbs':
          node.Sample()
      if dumpstate:
        if method == 'gibbs':
          for node in self.node_list:
            node.pos_state.assign(node.neg_state)
        for net in self.feed_backward_net:
          net.ForwardPropagate(train=False)
        self.DumpModelState(i)

    # End of Inference.
    if method == 'gibbs':
      for node in self.node_list:
        node.pos_state.assign(node.neg_state)
        node.pos_sample.assign(node.neg_sample)

  def TrainOneBatch(self, step):
    losses1 = self.PositivePhase(train=True, step=step)
    if step == 0:
      if self.t_op.optimizer == deepnet_pb2.Operation.CD:
        for node in self.layer:
          node.TiePhases()
      elif self.t_op.optimizer == deepnet_pb2.Operation.PCD:
        for node in self.layer:
          node.InitializeNegPhase()
    losses2 = self.NegativePhase(step, train=True)
    losses1.extend(losses2)
    return losses1
  
  def EvaluateOneBatch(self):
    losses = self.PositivePhase(train=False, evaluate=True)
    return losses

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
                                dataset='test'):
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
      self.PositivePhase(train=False, evaluate=False)
      reprs = [l.state.asarray().T for l in layers]
      datawriter.Submit(reprs)
    sys.stdout.write('\n')
    datawriter.Commit()
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

  def DoInference(self, layername, numbatches, inputlayername=[], method='mf',
                  steps=0, validation=True):
    indices = range(numbatches * self.e_op.batchsize)
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
      self.InferOneBatch(layer_to_tap, inputlayer, steps, method)
    return self.rep, self.inputs

  def ReconstructOneBatch(self, layer, inputlayers):
    self.PositivePhase(train=False, evaluate=True)
    self.recon[self.recon_pos:self.recon_pos + self.e_op.batchsize,:] =\
      layer.state.asarray().T
    for i, l in enumerate(inputlayers):
      self.inputs[i][self.recon_pos:self.recon_pos + self.e_op.batchsize,:] =\
        l.data.asarray().T
    self.recon_pos += self.e_op.batchsize

  def InferOneBatch(self, layer, inputlayers, steps, method):
    self.Infer(steps, method)
    batchsize = self.e_op.batchsize
    self.rep[self.rep_pos:self.rep_pos + batchsize, :] =\
        layer.state.asarray().T
    for i, l in enumerate(inputlayers):
      self.inputs[i][self.rep_pos:self.rep_pos + batchsize,:] =\
          l.data.asarray().T
    self.rep_pos += batchsize

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
