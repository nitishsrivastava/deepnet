"""Implements a Deep Belief Network."""
from dbm import *

class DBN(DBM):
  @staticmethod
  def FuseLayers(node1, node2):
    print 'Fusing state of layer %s with data of %s' % (node1.name, node2.name)
    if hasattr(node2, 'data') and isinstance(node2.data, cm.CUDAMatrix):
      assert node1.state.shape == node2.data.shape,\
          'Shape mismatch cannot fuse %s and %s' % (
            node1.state.shape, node2.data.shape)
      node2.data.free_device_memory()
    node2.data = node1.state
    assert node2.data == node1.state

  @staticmethod
  def ConvertToFeedForward(model_name):
    """Concatenates lower level models and creates a feed forward net."""
    print 'Converting %s' % model_name
    model = util.ReadModel(model_name)
    ff = deepnet_pb2.Model()
    if model.lower_model:
      # Include the lower model in ff.
      for i, model_file in enumerate(model.lower_model):
        m = DBN.ConvertToFeedForward(model_file)
        ff.layer.extend(m.layer)
        ff.edge.extend(m.edge)
      # Include the top layer rbm in ff.
      for l in model.layer:
        if not l.is_input:
          ff.layer.extend([l])
      ff.edge.extend(model.edge)
    else:
      ff.CopyFrom(model)
    for e in ff.edge:
      e.directed = True
    ff.model_type = deepnet_pb2.Model.FEED_FORWARD_NET
    for l in ff.layer:
      print 'Layer %s is_input %s' % (l.name, l.is_input)
    print 'Conversion complete'
    return ff

  @staticmethod
  def ConvertToFeedBackward(model_name):
    """Creates a feed forward net for passing data back."""
    print 'Converting %s' % model_name
    model = util.ReadModel(model_name)
    ff = deepnet_pb2.Model()
    for l in model.layer:
      l.is_input = not l.is_input
      ff.layer.extend([l])
    for e in model.edge:
      temp = e.node1
      e.node1 = e.node2
      e.node2 = temp
      for p in e.param:
        mat = util.ParameterAsNumpy(p)
        p.mat = util.NumpyAsParameter(mat.T)
        del p.dimensions[:]
        p.dimensions.append(mat.shape[1])
        p.dimensions.append(mat.shape[0])
      e.directed = True
      temp = e.up_factor
      e.up_factor = e.down_factor
      e.down_factor = temp
      ff.edge.extend([e])

    # Include the lower model in ff.
    for i, model_file in enumerate(model.lower_model):
      m = DBN.ConvertToFeedBackward(model_file)
      for l in m.layer:
        if not l.name in [x.name for x in ff.layer]:
          ff.layer.extend([l])
      ff.edge.extend(m.edge)

    ff.model_type = deepnet_pb2.Model.FEED_FORWARD_NET
    for l in ff.layer:
      print 'Layer %s is_input %s' % (l.name, l.is_input)
    print 'Conversion complete'
    return ff

  def DumpModelState(self, step):
    state_dict = dict([(node.name, node.state.asarray().T) for node in self.node_list])
    for net in self.feed_forward_net:
      for node in net.node_list:
        state_dict['forward_%s' % node.name] = node.state.asarray().T
    for net in self.feed_backward_net:
      for node in net.node_list:
        state_dict['backward_%s' % node.name] = node.state.asarray().T
    filename = '/ais/gobi3/u/nitish/flickr/states/%s_%.5d' % (self.net.name, step)
    print 'Dumping state at step %d to %s' % (step, filename)
    np.savez(filename, **state_dict)

  def SetUpData(self, *args):
    """Here we hook up the inputs of the DBN to the model below them."""
    self.feed_forward_net = []
    for model_file in self.net.lower_model:
      print 'Hooking up model %s' % (model_file) 
      model_pb = DBN.ConvertToFeedForward(model_file)
      assert model_pb.model_type == deepnet_pb2.Model.FEED_FORWARD_NET
      model = NeuralNet(model_pb, self.t_op, self.e_op)
      args = model.SetUpTrainer(*args)
      add_to_ff = False
      for node in self.input_datalayer:
        assert node.proto.HasField('data_field')
        for layer in model.layer:
          if layer.name == node.proto.data_field.layer_name:
            DBN.FuseLayers(layer, node)
            add_to_ff = True
      if add_to_ff:
        self.feed_forward_net.append(model)
    args = super(DBN, self).SetUpData(*args)
    return args

  def SetUpBackwardData(self, tap_down_from=[]):
    """Here we hook up the inputs of the DBN to the model below them.
    Args:
      tap_down_from: Hook these layers to the feed_backward_nets. The model
      already links the input layers. This argument is used for any other layers
      that one way want to tap down through a net.
    """
    self.feed_backward_net = []
    tap_down_from.extend(self.input_datalayer)
    tap_down_from = list(set(tap_down_from))
    for model_file in self.net.lower_model:
      print 'Hooking up model %s' % (model_file) 
      model_pb = DBN.ConvertToFeedBackward(model_file)
      assert model_pb.model_type == deepnet_pb2.Model.FEED_FORWARD_NET
      model = NeuralNet(model_pb, self.t_op, self.e_op)
      model.SetUpTrainer()
      add_to_ff = False
      for node in tap_down_from:
        assert node.proto.HasField('data_field')
        for layer in model.layer:
          if layer.name == node.proto.data_field.layer_name:
            DBN.FuseLayers(node, layer)
            add_to_ff = True
      if add_to_ff:
        self.feed_backward_net.append(model)

  def SetUpTrainer(self, *args):
    """Load the model, setup the data, set the stopping conditions."""
    self.LoadModelOnGPU()
    self.PrintNetwork()
    args = self.SetUpData(*args)
    if self.feed_forward_net:
      self.train_stop_steps = self.feed_forward_net[0].train_stop_steps
      self.validation_stop_steps = self.feed_forward_net[0].validation_stop_steps
      self.test_stop_steps = self.feed_forward_net[0].test_stop_steps
    self.eval_now_steps = self.t_op.eval_after
    self.save_now_steps = self.t_op.checkpoint_after
    self.show_now_steps = self.t_op.show_after
    return args

  def TrainOneBatch(self, step):
    # Get the data through the lower model(s).
    for net in self.feed_forward_net:
      net.ForwardPropagate(train=True)
    # Do what we usually do with DBMs/RBMs.
    return super(DBN, self).TrainOneBatch(step)

  def EvaluateOneBatch(self):
    # Get the data through the lower model(s).
    for net in self.feed_forward_net:
      net.ForwardPropagate(train=False)
    # Do what we usually do with DBMs/RBMs.
    return super(DBN, self).EvaluateOneBatch()

  def GetTrainBatch(self):
    """Ask the feed forward nets to get the training data."""
    for net in self.feed_forward_net:
      net.GetTrainBatch()
    for layer in self.input_datalayer:
      if layer.train_data_handler is not None:
        layer.GetTrainData()
    for layer in self.output_datalayer:
      if layer.train_data_handler is not None:
        layer.GetTrainData()

  def GetValidationBatch(self):
    """Ask the feed forward nets to get the validation data."""
    for net in self.feed_forward_net:
      net.GetValidationBatch()
    for layer in self.input_datalayer:
      if layer.validation_data_handler is not None:
        layer.GetValidationData()
    for layer in self.output_datalayer:
      if layer.validation_data_handler is not None:
        layer.GetValidationData()

  def GetTestBatch(self):
    """Ask the feed forward nets to get the test data."""
    for net in self.feed_forward_net:
      net.GetTestBatch()
    for layer in self.input_datalayer:
      if layer.test_data_handler is not None:
        layer.GetTestData()
    for layer in self.output_datalayer:
      if layer.test_data_handler is not None:
        layer.GetTestData()


  def DoInference(self, layername, numbatches, inputlayername=[], method='mf',
                  steps=0, validation=True):
    step = 0
    indices = range(numbatches * self.e_op.batchsize)
    self.rep_pos = 0
    inputlayer = []
    self.inputs = []
    layer_to_tap = self.GetLayerByName(layername, down=True)
    self.rep = np.zeros((numbatches * self.e_op.batchsize,
                         layer_to_tap.state.shape[0]))
    for i, lname in enumerate(inputlayername):
      l = self.GetLayerByName(lname)
      inputlayer.append(l)
      self.inputs.append(np.zeros((numbatches * self.e_op.batchsize,
                                   l.state.shape[0])))
    if validation:
      datagetter = self.GetValidationBatch
      #datagetter = self.GetTrainBatch
    else:
      datagetter = self.GetTestBatch
    for batch in range(numbatches):
      datagetter()
      self.InferOneBatch(layer_to_tap, inputlayer, steps, method, mf_for_last,
                         get_states_after_gibbs)
    return self.rep, self.inputs

  def GetAllRepresentations(self, numbatches, validation=True):
    if validation:
      datagetter = self.GetValidationBatch
    else:
      datagetter = self.GetTestBatch
    rep_list = []
    names = []
    for node in self.node_list:
      if not node.is_input:
        rep_list.append(np.zeros((numbatches * node.state.shape[1],
                                  node.state.shape[0]), dtype='float32'))
        names.append(node.name)
    for net in self.feed_forward_net:
      for node in net.node_list:
        rep_list.append(np.zeros((numbatches * node.state.shape[1],
                                  node.state.shape[0]), dtype='float32'))
        names.append(node.name)

    for batch in range(numbatches):
      datagetter()
      for net in self.feed_forward_net:
        net.ForwardPropagate(train=False)
      self.PositivePhase(train=False, evaluate=False)
      i = 0
      for node in self.node_list:
        if not node.is_input:
          rep_list[i][batch*node.batchsize:(batch+1)*node.batchsize,:] =\
              node.state.asarray().T
          i += 1
      for net in self.feed_forward_net:
        for node in net.node_list:
          rep_list[i][batch*node.batchsize:(batch+1)*node.batchsize,:] =\
              node.state.asarray().T
          i += 1
    return dict(zip(names, rep_list))

  def ReconstructOneBatch(self, layer, inputlayer=[]):
    for net in self.feed_forward_net:
      net.ForwardPropagate(train=False)
    self.PositivePhase(train=False, evaluate=True)
    for net in self.feed_backward_net:
      net.ForwardPropagate(train=False)
    self.recon[self.recon_pos:self.recon_pos + self.e_op.batchsize,:] =\
        layer.state.asarray().T
    for i, l in enumerate(inputlayer):
      self.inputs[i][self.recon_pos:self.recon_pos + self.e_op.batchsize,:] =\
          l.data.asarray().T
    self.recon_pos += self.e_op.batchsize

  def GetRepresentationOneBatch(self, layer, inputlayer=[]):
    for net in self.feed_forward_net:
      net.ForwardPropagate(train=False)
    self.PositivePhase(train=False, evaluate=False)
    self.rep[self.rep_pos:self.rep_pos + self.e_op.batchsize,:] =\
        layer.state.asarray().T
    for i, l in enumerate(inputlayer):
      self.inputs[i][self.rep_pos:self.rep_pos + self.e_op.batchsize,:] =\
          l.data.asarray().T
    self.rep_pos += self.e_op.batchsize

  def InferOneBatch(self, layer, inputlayers, steps, method, dumpstate=False):
    for net in self.feed_forward_net:
      net.ForwardPropagate(train=False)
    self.Infer(steps, method, mf_for_last)
    for net in self.feed_backward_net:
      net.ForwardPropagate(train=False)
    if dumspstate:
      self.DumpModelState(steps)
    self.rep[self.rep_pos:self.rep_pos + self.e_op.batchsize, :] =\
        layer.state.asarray().T
    for i, l in enumerate(inputlayers):
      self.inputs[i][self.rep_pos:self.rep_pos + self.e_op.batchsize,:] =\
        l.state.asarray().T
    self.rep_pos += self.e_op.batchsize

  def GetLayerByName(self, layername, down=False):
    if down:
      return self.GetLayerByNameFeedBackward(layername)
    else:
      return self.GetLayerByNameFeedForward(layername)

  def GetLayerByNameFeedForward(self, layername):
    layer = None
    try:
      layer = next(layer for layer in self.layer if layer.name == layername)
    except StopIteration:
      for ff in self.feed_forward_net:
        try:
          layer = next(layer for layer in ff.layer if layer.name == layername)
        except StopIteration:
          pass
    if layer is None:
      print 'No such layer %s' % layername
    return layer

  def GetLayerByNameFeedBackward(self, layername):
    layer = None
    try:
      layer = next(layer for layer in self.layer if layer.name == layername)
    except StopIteration:
      for ff in self.feed_backward_net:
        try:
          layer = next(layer for layer in ff.layer if layer.name == layername)
        except StopIteration:
          pass
    if layer is None:
      print 'No such layer %s' % layername
    return layer
