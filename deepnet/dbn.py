"""Implements a Deep Belief Network."""
from dbm import *

class DBN(DBM):

  def __init__(self, net, t_op=None, e_op=None):
    rbm, upward_net, downward_net, junction_layers = DBN.SplitDBN(net)
    self.rbm = DBM(rbm, t_op, e_op)
    self.upward_net = NeuralNet(upward_net, t_op, e_op)
    self.downward_net = NeuralNet(downward_net, t_op, e_op)
    self.junction_layers = junction_layers
    self.net = self.rbm.net
    self.t_op = self.rbm.t_op
    self.e_op = self.rbm.e_op
    self.verbose = self.rbm.verbose
    self.batchsize = self.t_op.batchsize

  def CopyModelToCPU(self):
    self.rbm.CopyModelToCPU()

  def DeepCopy(self):
    return CopyModel(self.rbm.net)

  def Show(self):
    """Visualize the state of the layers and edges in the network."""
    self.rbm.Show()
    self.upward_net.Show()
    self.downward_net.Show()

  def PrintNetwork(self):
    print 'RBM:'
    self.rbm.PrintNetwork()
    print 'Up:'
    self.upward_net.PrintNetwork()
    print 'Down:'
    self.downward_net.PrintNetwork()

  def ExchangeGlobalInfo(self):
    for layer in self.rbm.layer:
      layer.GetGlobalInfo(self)
    for edge in self.rbm.edge:
      edge.GetGlobalInfo(self)

  @staticmethod
  def SplitDBN(net):
    #net = ReadModel(dbn_file)
    rbm = deepnet_pb2.Model()
    rbm.CopyFrom(net)
    rbm.name = '%s_rbm' % net.name
    rbm.model_type = deepnet_pb2.Model.DBM
    
    directed_edges = []
    undirected_edges = []
    layer1 = set()  # Layers that are touched by directed edges.
    layer2 = set()  # Layers that are touched by undirected edges.
    for e in net.edge:
      if e.directed:
        directed_edges.append(e)
        layer1.add(e.node1)
        layer1.add(e.node2)
      else:
        undirected_edges.append(e)
        layer2.add(e.node1)
        layer2.add(e.node2)

    junction_layers = list(layer1.intersection(layer2))
    
    # CONTRUCT RBM.
    del rbm.edge[:]
    for e in undirected_edges:
      rbm.edge.extend([e])

    del rbm.layer[:]
    for node in list(layer2):
      l = next(l for l in net.layer if l.name == node)
      layer = rbm.layer.add()
      layer.CopyFrom(l)
      if node in junction_layers:
        layer.is_input = True
        del layer.param[:]
        for p in l.param:
          if p.name == 'bias':
            continue
          elif p.name == 'bias_generative':
            p_copy = layer.param.add()
            p_copy.CopyFrom(p)
            p_copy.name = 'bias'
          else:
            layer.param.extend([p])

    # CONSTRUCT DOWNNARD NET.
    down_net = deepnet_pb2.Model()
    down_net.CopyFrom(net)
    down_net.name = '%s_downward_net' % net.name
    down_net.model_type = deepnet_pb2.Model.FEED_FORWARD_NET

    del down_net.edge[:]
    for e in directed_edges:
      down_net.edge.extend([e])

    del down_net.layer[:]
    for node in list(layer1):
      l = next(l for l in net.layer if l.name == node)
      layer_down = down_net.layer.add()
      layer_down.CopyFrom(l)
      if l.is_input:
        layer_down.is_input = False
      if node in junction_layers:
        layer_down.is_input = True
      del layer_down.param[:]
      for p in l.param:
        if p.name == 'bias':
          continue
        elif p.name == 'bias_generative':
          p_copy = layer_down.param.add()
          p_copy.CopyFrom(p)
          p_copy.name = 'bias'
        else:
          layer_down.param.extend([p])

    # CONSTRUCT UPWARD NET.
    up_net = deepnet_pb2.Model()
    up_net.CopyFrom(net)
    up_net.name = '%s_upward_net' % net.name
    up_net.model_type = deepnet_pb2.Model.FEED_FORWARD_NET
    del up_net.edge[:]
    for e in directed_edges:
      e_up = DBN.ReverseEdge(e)
      up_net.edge.extend([e_up])
    del up_net.layer[:]
    for node in list(layer1):
      l = next(l for l in net.layer if l.name == node)
      layer_up = up_net.layer.add()
      layer_up.CopyFrom(l)
      del layer_up.param[:]
      for p in l.param:
        if p.name == 'bias_generative':
          continue
        else:
          layer_up.param.extend([p])

    return rbm, up_net, down_net, junction_layers

  @staticmethod
  def ReverseEdge(e):
    rev_e = deepnet_pb2.Edge()
    rev_e.CopyFrom(e)
    rev_e.node1 = e.node2
    rev_e.node2 = e.node1
    rev_e.up_factor = e.down_factor
    rev_e.down_factor = e.up_factor
    for p in rev_e.param:
      if p.name == 'weight':
        if p.initialization == deepnet_pb2.Parameter.PRETRAINED:
          p.transpose_pretrained = not p.transpose_pretrained
        elif p.mat:
          mat = ParameterAsNumpy(p).T
          p.mat = NumpyAsParameter(mat)
          del p.dimensions
          for dim in mat.shape:
            p.dimensions.add(dim)
    return rev_e

  def LoadModelOnGPU(self, *args, **kwargs):
    self.rbm.LoadModelOnGPU(*args, **kwargs)
    self.upward_net.LoadModelOnGPU(*args, **kwargs)
    self.downward_net.LoadModelOnGPU(*args, **kwargs)
    self.TieUpNets()

  def TieUpNets(self):
    # Tie up nets.
    for layer_name in self.junction_layers:
      rbm_layer = next(l for l in self.rbm.layer if l.name == layer_name)
      up_layer = next(l for l in self.upward_net.layer if l.name == layer_name)
      down_layer = next(l for l in self.downward_net.layer if l.name == layer_name)
      rbm_layer.data = up_layer.state
      down_layer.data = rbm_layer.state

  def ResetBatchsize(self, batchsize):
    self.batchsize = batchsize
    self.rbm.ResetBatchsize(batchsize)
    self.upward_net.ResetBatchsize(batchsize)
    self.downward_net.ResetBatchsize(batchsize)
    self.TieUpNets()

  def SetUpData(self, *args, **kwargs):
    self.upward_net.SetUpData(*args, **kwargs)
    self.train_data_handler = self.upward_net.train_data_handler
    self.validation_data_handler = self.upward_net.validation_data_handler
    self.test_data_handler = self.upward_net.test_data_handler

  def GetBatch(self, handler=None):
    if handler:
      data_list = handler.Get()
      if data_list[0].shape[1] != self.batchsize:
        self.ResetBatchsize(data_list[0].shape[1])
      for i, layer in enumerate(self.upward_net.datalayer):
        layer.SetData(data_list[i])
    for layer in self.upward_net.tied_datalayer:
      layer.SetData(layer.tied_to.data)

  def TrainOneBatch(self, step):
    self.upward_net.ForwardPropagate(train=True, step=step)
    return self.rbm.TrainOneBatch(step)
 
  def PositivePhase(self, train=False, evaluate=False, step=0):
    self.upward_net.ForwardPropagate(train=train, step=step)
    return self.rbm.PositivePhase(train=train, evaluate=evaluate, step=step)
    #self.downward_net.ForwardPropagate(train=train, step=step)

  def NegativePhase(self, *args, **kwargs):
    return self.rbm.NegativePhase(*args, **kwargs)

  def Inference(self, steps, layernames, unclamped_layers, output_dir, memory='1G', dataset='test', method='gibbs'):
    layers_to_infer = [self.GetLayerByName(l, down=True) for l in layernames]
    layers_to_unclamp = [self.GetLayerByName(l) for l in unclamped_layers]

    numdim_list = [layer.state.shape[0] for layer in layers_to_infer]
    upward_net_unclamped_inputs = []
    for l in layers_to_unclamp:
      l.is_input = False
      l.is_initialized = True
      if l in self.rbm.layer:
        self.rbm.pos_phase_order.append(l)
      else:
        upward_net_unclamped_inputs.append(l)

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
      for l in upward_net_unclamped_inputs:
        l.data.assign(0)
      self.upward_net.ForwardPropagate()
      for node in self.rbm.node_list:
        if node.is_input or node.is_initialized:
          node.GetData()
          if gibbs:
            node.sample.assign(node.state)
        else:
          node.ResetState(rand=False)
      for i in range(steps):
        for node in self.rbm.pos_phase_order:
          self.ComputeUp(node, use_samples=gibbs)
          if gibbs:
            node.Sample()
      self.downward_net.ForwardPropagate()
      output = [l.state.asarray().T for l in layers_to_infer]
      dw.Submit(output)
    sys.stdout.write('\n')
    size = dw.Commit()
    return size[0]

  def GetLayerByName(self, layername, down=False):
    layer = self.rbm.GetLayerByName(layername)
    if layer is None:
      if down:
        layer = self.downward_net.GetLayerByName(layername)
      else:
        layer = self.upward_net.GetLayerByName(layername)
    return layer
