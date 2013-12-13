"""Implements a feed-forward neural net."""
import gzip
import logging
import sys
import time
from google.protobuf import text_format

from datahandler import *
from convolutions import *
from edge import *
from layer import *
from util import *
from logistic_layer import *
from tanh_layer import *
from relu_layer import *
from smooth_relu_layer import *
from linear_layer import *
from softmax_layer import *
from replicated_softmax_layer import *
from cos_layer import *
from sin_layer import *
from transfer_edge import *
from soft_transfer_edge import *

class NeuralNet(object):

  def __init__(self, net, t_op=None, e_op=None):
    self.net = None
    if isinstance(net, deepnet_pb2.Model):
      self.net = net
    elif isinstance(net, str) or isinstance(net, unicode):
      self.net = ReadModel(net)
    self.t_op = None
    if isinstance(t_op, deepnet_pb2.Operation):
      self.t_op = t_op
    elif isinstance(t_op, str) or isinstance(net, unicode):
      self.t_op = ReadOperation(t_op)
    self.e_op = None
    if isinstance(e_op, deepnet_pb2.Operation):
      self.e_op = e_op
    elif isinstance(e_op, str) or isinstance(net, unicode):
      self.e_op = ReadOperation(e_op)
    cm.CUDAMatrix.init_random(self.net.seed)
    np.random.seed(self.net.seed)
    self.data = None
    self.layer = []
    self.edge = []
    self.input_datalayer = []
    self.output_datalayer = []
    self.datalayer = []
    self.tied_datalayer = []
    self.unclamped_layer = []
    self.verbose = False
    self.batchsize = 0
    if self.t_op:
      self.verbose = self.t_op.verbose
      self.batchsize = self.t_op.batchsize
    elif self.e_op:
      self.verbose = self.e_op.verbose
      self.batchsize = self.e_op.batchsize
    self.train_stop_steps = sys.maxint

  def PrintNetwork(self):
    for layer in self.layer:
      print layer.name
      layer.PrintNeighbours()

  def DeepCopy(self):
    return CopyModel(self.net)

  def LoadModelOnGPU(self, batchsize=-1):
    """Load the model on the GPU."""
    if batchsize < 0:
      if self.t_op:
        batchsize=self.t_op.batchsize
      else:
        batchsize=self.e_op.batchsize

    for layer in self.net.layer:
      layer.hyperparams.MergeFrom(LoadMissing(layer.hyperparams,
                                              self.net.hyperparams))
      if not layer.prefix:
        layer.prefix = self.net.prefix
      tied_to = None
      if layer.tied:
        tied_to = next(l for l in self.layer if l.name == layer.tied_to)
      self.layer.append(CreateLayer(Layer, layer, self.t_op, tied_to=tied_to))

    for edge in self.net.edge:
      hyp = deepnet_pb2.Hyperparams()
      hyp.CopyFrom(self.net.hyperparams)
      hyp.MergeFrom(edge.hyperparams)
      edge.hyperparams.MergeFrom(hyp)
      try:
        node1 = next(layer for layer in self.layer if layer.name == edge.node1)
      except StopIteration:
        print edge.node1, [l.name for l in self.layer]
      node2 = next(layer for layer in self.layer if layer.name == edge.node2)
      if not edge.prefix:
        edge.prefix = self.net.prefix
      tied_to = None
      if edge.tied:
        tied_to = next(e for e in self.edge if e.node1.name == edge.tied_to_node1 and e.node2.name == edge.tied_to_node2)
      self.edge.append(CreateEdge(Edge, edge, node1, node2, self.t_op, tied_to=tied_to))

    self.input_datalayer = [node for node in self.layer if node.is_input]
    self.output_datalayer = [node for node in self.layer if node.is_output]
    self.node_list = self.Sort()

  def ExchangeGlobalInfo(self):
    for layer in self.layer:
      layer.GetGlobalInfo(self)
    for edge in self.edge:
      edge.GetGlobalInfo(self)

  def Sort(self):
    """Topological sort."""
    node_list = []
    S = [node for node in self.layer if not node.incoming_neighbour]
    while S:
      n = S.pop()
      node_list.append(n)
      for m in n.outgoing_edge:
        if m.marker == 0:
          m.marker = 1
          if reduce(lambda a, edge: a and edge.marker == 1,
                    m.node2.incoming_edge, True):
            S.append(m.node2)
    if reduce(lambda a, edge: a and edge.marker == 1, self.edge, True):
      if self.verbose:
        print 'Fprop Order:'
        for node in node_list:
          print node.name
    else:
      raise Exception('Invalid net for backprop. Cycle exists.')
    return node_list

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
        perf.sparsity = layer.dimsize.sum() / layer.dimsize.shape[0]

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
    if layer.is_output:
      loss = layer.GetLoss(get_deriv=True)
    else:
      loss = None
      if layer.hyperparams.sparsity:
        sparsity_gradient = layer.GetSparsityGradient()
        layer.deriv.add_col_vec(sparsity_gradient)
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

  def AccumulateDeriv(self, layer, edge, deriv):
    """Accumulate the derivative w.r.t the outputs of this layer.

    A layer needs to compute derivatives w.r.t its outputs. These outputs may
    have been connected to lots of other nodes through outgoing edges.
    This method adds up the derivatives contributed by each outgoing edge.
    It gets derivatives w.r.t the inputs at the other end of its outgoing edge.
    Args:
      edge: The edge which is sending the derivative.
      deriv: The derivative w.r.t the inputs at the other end of this edge.
    """
    if layer.is_input or edge.proto.block_gradient:
      return
    if layer.dirty:  # If some derivatives have already been received.
      layer.deriv.add_dot(edge.params['weight'], deriv)
    else:  # Receiving derivative for the first time.
      cm.dot(edge.params['weight'], deriv, target=layer.deriv)
      layer.dirty = True

  def UpdateEdgeParams(self, edge, deriv, step):
    """ Update the parameters associated with this edge.

    Update the weights and associated parameters.
    Args:
      deriv: Gradient w.r.t the inputs at the outgoing end.
      step: Training step.
    """
    numcases = edge.node1.batchsize
    if edge.conv or edge.local:
      ConvOuter(edge, edge.temp)
      edge.gradient.add_mult(edge.temp, mult=1.0/numcases)
    else:
      edge.gradient.add_dot(edge.node1.state, deriv.T, mult=1.0/numcases)
    if edge.tied_to:
      edge.tied_to.gradient.add(edge.gradient)
      edge.gradient.assign(0)
      edge = edge.tied_to
    edge.num_grads_received += 1
    if edge.num_grads_received == edge.num_shares:
      edge.Update('weight', step)

  def UpdateLayerParams(self, layer, step):
    """ Update the parameters associated with this layer.
    Update the bias.
    Args:
      step: Training step.
    """
    layer.gradient.add_sums(layer.deriv, axis=1, mult=1.0 / layer.batchsize)
    if layer.tied_to:
      layer.tied_to.gradient.add(layer.gradient)
      layer.gradient.assign(0)
      layer = layer.tied_to
    layer.num_grads_received += 1
    if layer.num_grads_received == layer.num_shares:
      layer.Update('bias', step, no_reg=True)  # By default, do not regularize bias.

  def ForwardPropagate(self, train=False, step=0):
    """Do a forward pass through the network.

    Args:
      train: True if the forward pass is done during training, False during
        evaluation.
      step: Training step.
    """
    losses = []
    for node in self.node_list:
      loss = self.ComputeUp(node, train, step, self.train_stop_steps)
      if loss:
        losses.append(loss)
    return losses

  def BackwardPropagate(self, step):
    """Backprop through the network.

    Args:
      step: Training step.
    """
    losses = []
    for node in reversed(self.node_list):
      loss = self.ComputeDown(node, step)
      if loss:
        losses.append(loss)
    return losses

  def TrainOneBatch(self, step):
    """Train once on one mini-batch.

    Args:
      step: Training step.
    Returns:
      List of losses incurred at each output layer.
    """
    losses1 = self.ForwardPropagate(train=True)
    losses2 = self.BackwardPropagate(step)
    losses1.extend(losses2)
    return losses1

  def EvaluateOneBatch(self):
    """Evaluate one mini-batch."""
    losses = self.ForwardPropagate()
    losses.extend([node.GetLoss() for node in self.output_datalayer])
    return losses

  def Evaluate(self, validation=True, collect_predictions=False):
    """Evaluate the model.
    Args:
      validation: If True, evaluate on the validation set,
        else evaluate on test set.
      collect_predictions: If True, collect the predictions.
    """
    step = 0
    stats = []
    if validation:
      stopcondition = self.ValidationStopCondition
      stop = stopcondition(step)
      if stop or self.validation_data_handler is None:
        return
      datagetter = self.GetValidationBatch
      prefix = 'V'
      stats_list = self.net.validation_stats
      num_batches = self.validation_data_handler.num_batches
    else:
      stopcondition = self.TestStopCondition
      stop = stopcondition(step)
      if stop or self.test_data_handler is None:
        return
      datagetter = self.GetTestBatch
      prefix = 'E'
      stats_list = self.net.test_stats
      num_batches = self.test_data_handler.num_batches
    if collect_predictions:
      output_layer = self.output_datalayer[0]
      collect_pos = 0
      batchsize = output_layer.batchsize
      numdims = output_layer.state.shape[0]
      predictions = np.zeros((batchsize * num_batches, numdims))
      targets = np.zeros(predictions.shape)
    while not stop:
      datagetter()
      losses = self.EvaluateOneBatch()
      if collect_predictions:
        predictions[collect_pos:collect_pos + batchsize] = \
            output_layer.state.asarray().T
        targets[collect_pos:collect_pos + batchsize] = \
            output_layer.data.asarray().T
        collect_pos += batchsize

      if stats:
        for loss, acc in zip(losses, stats):
          Accumulate(acc, loss)
      else:
        stats = losses
      step += 1
      stop = stopcondition(step)
    if collect_predictions and stats:
      predictions = predictions[:collect_pos]
      targets = targets[:collect_pos]
      MAP, prec50, MAP_list, prec50_list = self.ComputeScore(predictions, targets)
      stat = stats[0]
      stat.MAP = MAP
      stat.prec50 = prec50
      for m in MAP_list:
        stat.MAP_list.extend([m])
      for m in prec50_list:
        stat.prec50_list.extend([m])
    for stat in stats:
      sys.stdout.write(GetPerformanceStats(stat, prefix=prefix))
    stats_list.extend(stats)


  def ScoreOneLabel(self, preds, targets):
    """Computes Average precision and precision at 50."""
    targets_sorted = targets[(-preds.T).argsort().flatten(),:]
    cumsum = targets_sorted.cumsum()
    prec = cumsum / np.arange(1.0, 1 + targets.shape[0])
    total_pos = float(sum(targets))
    if total_pos == 0:
      total_pos = 1e-10
    recall = cumsum / total_pos
    ap = np.dot(prec, targets_sorted) / total_pos
    prec50 = prec[50]
    return ap, prec50

  def ComputeScore(self, preds, targets):
    """Computes Average precision and precision at 50."""
    assert preds.shape == targets.shape
    numdims = preds.shape[1]
    ap = 0
    prec = 0
    ap_list = []
    prec_list = []
    for i in range(numdims):
      this_ap, this_prec = self.ScoreOneLabel(preds[:,i], targets[:,i])
      ap_list.append(this_ap)
      prec_list.append(this_prec)
      ap += this_ap
      prec += this_prec
    ap /= numdims
    prec /= numdims
    return ap, prec, ap_list, prec_list

  def WriteRepresentationToDisk(self, layernames, output_dir, memory='1G',
                                dataset='test', drop=False):
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
      self.ForwardPropagate(train=drop)
      reprs = [l.state.asarray().T for l in layers]
      datawriter.Submit(reprs)
    sys.stdout.write('\n')
    return datawriter.Commit()

  def TrainStopCondition(self, step):
    return step >= self.train_stop_steps

  def ValidationStopCondition(self, step):
    return step >= self.validation_stop_steps

  def TestStopCondition(self, step):
    return step >= self.test_stop_steps

  def EvalNow(self, step):
    return step % self.eval_now_steps == 0

  def SaveNow(self, step):
    return step % self.save_now_steps == 0

  def ShowNow(self, step):
    return self.show_now_steps > 0 and step % self.show_now_steps == 0

  def GetLayerByName(self, layername, down=False):
    try:
      l = next(l for l in self.layer if l.name == layername)
    except StopIteration:
      l = None
    return l

  def CopyModelToCPU(self):
    for layer in self.layer:
      layer.SaveParameters()
    for edge in self.edge:
      edge.SaveParameters()

  def ResetBatchsize(self, batchsize):
    self.batchsize = batchsize
    for layer in self.layer:
      layer.AllocateBatchsizeDependentMemory(batchsize)
    for edge in self.edge:
      edge.AllocateBatchsizeDependentMemory()

  def GetBatch(self, handler=None):
    if handler:
      data_list = handler.Get()
      if data_list[0].shape[1] != self.batchsize:
        self.ResetBatchsize(data_list[0].shape[1])
      for i, layer in enumerate(self.datalayer):
        layer.SetData(data_list[i])
    for layer in self.tied_datalayer:
      data = layer.data_tied_to.data
      if data.shape[1] != self.batchsize:
        self.ResetBatchsize(data.shape[1])
      layer.SetData(data)

  def GetTrainBatch(self):
    self.GetBatch(self.train_data_handler)

  def GetValidationBatch(self):
    self.GetBatch(self.validation_data_handler)

  def GetTestBatch(self):
    self.GetBatch(self.test_data_handler)

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
      if data_field.tied:
        self.tied_datalayer.append(node)
        node.data_tied_to = next(l for l in self.datalayer\
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

  def SetUpTrainer(self):
    """Load the model, setup the data, set the stopping conditions."""
    self.LoadModelOnGPU()
    if self.verbose:
      self.PrintNetwork()
    self.SetUpData()
    if self.t_op.stopcondition.all_processed:
      num_steps = self.train_data_handler.num_batches
    else:
      num_steps = self.t_op.stopcondition.steps
    self.train_stop_steps = num_steps
    if self.e_op.stopcondition.all_processed and self.validation_data_handler:
      num_steps = self.validation_data_handler.num_batches
    else:
      num_steps = self.e_op.stopcondition.steps
    self.validation_stop_steps = num_steps
    if self.e_op.stopcondition.all_processed and self.test_data_handler:
      num_steps = self.test_data_handler.num_batches
    else:
      num_steps = self.e_op.stopcondition.steps
    self.test_stop_steps = num_steps

    self.eval_now_steps = self.t_op.eval_after
    self.save_now_steps = self.t_op.checkpoint_after
    self.show_now_steps = self.t_op.show_after

    self.ExchangeGlobalInfo()

  def Show(self):
    """Visualize the state of the layers and edges in the network."""
    for layer in self.layer:
      layer.Show()
    for edge in self.edge:
      edge.Show()

  def Train(self):
    """Train the model."""
    assert self.t_op is not None, 't_op is None.'
    assert self.e_op is not None, 'e_op is None.'
    self.SetUpTrainer()
    step = self.t_op.current_step
    stop = self.TrainStopCondition(step)
    stats = []

    collect_predictions = False
    try:
      p = self.output_datalayer[0].proto.performance_stats
      if p.compute_MAP or p.compute_prec50:
        collect_predictions = True
    except Exception as e:
      pass
    select_model_using_error = self.net.hyperparams.select_model_using_error
    select_model_using_acc = self.net.hyperparams.select_model_using_acc
    select_model_using_map = self.net.hyperparams.select_model_using_map
    select_best = select_model_using_error or select_model_using_acc or select_model_using_map
    if select_best:
      best_valid_error = float('Inf')
      test_error = float('Inf')
      best_net = self.DeepCopy()

    dump_best = False
    while not stop:
      sys.stdout.write('\rTrain Step: %d' % step)
      sys.stdout.flush()
      self.GetTrainBatch()
      losses = self.TrainOneBatch(step)
      if stats:
        for acc, loss in zip(stats, losses):
          Accumulate(acc, loss)
      else:
        stats = losses
      step += 1
      if self.ShowNow(step):
        self.Show()
      if self.EvalNow(step):
        # Print out training stats.
        sys.stdout.write('\rStep %d ' % step)
        for stat in stats:
          sys.stdout.write(GetPerformanceStats(stat, prefix='T'))
        self.net.train_stats.extend(stats)
        stats = []
        # Evaluate on validation set.
        self.Evaluate(validation=True, collect_predictions=collect_predictions)
        # Evaluate on test set.
        self.Evaluate(validation=False, collect_predictions=collect_predictions)
        if select_best:
          valid_stat = self.net.validation_stats[-1]
          if len(self.net.test_stats) > 1:
            test_stat = self.net.test_stats[-1]
          else:
            test_stat = valid_stat
          if select_model_using_error:
            valid_error = valid_stat.error / valid_stat.count
            _test_error = test_stat.error / test_stat.count
          elif select_model_using_acc:
            valid_error = 1 - float(valid_stat.correct_preds) / valid_stat.count
            _test_error = 1 - float(test_stat.correct_preds) / test_stat.count
          elif select_model_using_map:
            valid_error = 1 - valid_stat.MAP
            _test_error = 1 - test_stat.MAP
          if valid_error < best_valid_error:
            best_valid_error = valid_error
            test_error = _test_error
            dump_best = True
            self.CopyModelToCPU()
            self.t_op.current_step = step
            self.net.best_valid_stat.CopyFrom(valid_stat)
            self.net.train_stat_es.CopyFrom(self.net.train_stats[-1])
            self.net.test_stat_es.CopyFrom(test_stat)
            best_net = self.DeepCopy()
            best_t_op = CopyOperation(self.t_op)
        #for e in self.edge:
        #  sys.stdout.write(' %s %.3f' % (e.name, e.params['weight'].euclid_norm()))
        sys.stdout.write('\n')
      if self.SaveNow(step):
        self.t_op.current_step = step
        self.CopyModelToCPU()
        util.WriteCheckpointFile(self.net, self.t_op)
        if dump_best:
          dump_best = False
          if select_model_using_error:
            print 'Best valid error : %.4f Test error %.4f' % (best_valid_error, test_error)
          elif select_model_using_acc:
            print 'Best valid acc : %.4f Test acc %.4f' % (1-best_valid_error, 1-test_error)
          elif select_model_using_map:
            print 'Best valid MAP : %.4f Test MAP %.4f' % (1-best_valid_error, 1-test_error)

          util.WriteCheckpointFile(best_net, best_t_op, best=True)

      stop = self.TrainStopCondition(step)
