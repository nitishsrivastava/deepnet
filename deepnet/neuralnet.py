"""Implements a feed-forward neural net."""
import gzip
import logging
import sys
import time

from datahandler import *
from edge import *
from google.protobuf import text_format
from layer import *
from util import *

def GetPerformanceStats(stat, prefix=''):
  s = ''
  if stat.compute_cross_entropy:
    s += ' %s_CE: %.3f' % (prefix, stat.cross_entropy / stat.count)
  if stat.compute_correct_preds:
    s += ' %s_Acc: %.3f (%d/%d)' % (
      prefix, stat.correct_preds/stat.count, stat.correct_preds, stat.count)
  if stat.compute_error:
    s += ' %s_E: %.5f' % (prefix, stat.error / stat.count)
  if stat.compute_MAP and prefix != 'T':
    s += ' %s_MAP: %.3f' % (prefix, stat.MAP)
  if stat.compute_prec50 and prefix != 'T':
    s += ' %s_prec50: %.3f' % (prefix, stat.prec50)
  if stat.compute_sparsity:
    s += ' %s_sp: %.3f' % (prefix, stat.sparsity / stat.count)
  return s

def Accumulate(acc, perf):
 acc.count += perf.count
 acc.cross_entropy += perf.cross_entropy
 acc.error += perf.error
 acc.correct_preds += perf.correct_preds
 acc.sparsity += perf.sparsity

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
    self.SetLayerAndEdgeClass()
    self.verbose = False
    self.batchsize = 0
    if self.t_op:
      self.verbose = self.t_op.verbose
      self.batchsize = self.t_op.batchsize
    elif self.e_op:
      self.verbose = self.e_op.verbose
      self.batchsize = self.e_op.batchsize
    self.train_stop_steps = sys.maxint

  def SetLayerAndEdgeClass(self):
    self.LayerClass = Layer
    self.EdgeClass = Edge

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
      hyp = deepnet_pb2.Hyperparams()
      hyp.CopyFrom(self.net.hyperparams)
      hyp.MergeFrom(layer.hyperparams)
      layer.hyperparams.MergeFrom(hyp)
      self.layer.append(self.LayerClass(layer, batchsize))

    for edge in self.net.edge:
      hyp = deepnet_pb2.Hyperparams()
      hyp.CopyFrom(self.net.hyperparams)
      hyp.MergeFrom(edge.hyperparams)
      edge.hyperparams.MergeFrom(hyp)
      node1 = next(layer for layer in self.layer if layer.name == edge.node1)
      node2 = next(layer for layer in self.layer if layer.name == edge.node2)
      self.edge.append(self.EdgeClass(edge, node1, node2))

    self.input_datalayer = [node for node in self.layer if node.is_input]
    self.output_datalayer = [node for node in self.layer if node.is_output]
    self.node_list = self.Sort()

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

  def ForwardPropagate(self, train=False, step=0):
    """Do a forward pass through the network.

    Args:
      train: True if the forward pass is done during training, False during
        evaluation.
      step: Training step.
    """
    losses = []
    for node in self.node_list:
      loss = node.ComputeUp(train, step, self.train_stop_steps)
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
      loss = node.ComputeDown(step)
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
      layer.SetData(layer.tied_to.data)

  def GetTrainBatch(self):
    self.GetBatch(self.train_data_handler)

  def GetValidationBatch(self):
    self.GetBatch(self.validation_data_handler)

  def GetTestBatch(self):
    self.GetBatch(self.test_data_handler)

  def SetUpData(self, skip_outputs=False):
    """Setup the data."""
    hyp_list = []
    name_list = [[], [], []]
    for node in self.layer:
      if not (node.is_input or node.is_output):
        continue
      if skip_outputs and node.is_output:
        continue
      data_field = node.proto.data_field
      if data_field.tied:
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
      p = self.output_datalayer[0].proto.performance_statS
      if p.compute_MAP or p.compute_prec50:
        collect_predictions = True
    except Exception as e:
      pass
    select_model_using_error = self.net.hyperparams.select_model_using_error

    if select_model_using_error:
      best_error = float('Inf')
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
      if self.EvalNow(step):
        # Print out training stats.
        sys.stdout.write('\rStep %d ' % step)
        for stat in stats:
          sys.stdout.write(GetPerformanceStats(stat, prefix='T'))
        self.net.train_stats.extend(stats)
        stats = []
        # Evaluate on validation set.
        self.Evaluate(validation=True, collect_predictions=collect_predictions)
        if select_model_using_error:
          stat = self.net.validation_stats[-1]
          error = stat.error / stat.count
          if error < best_error:
            best_error = error
            dump_best = True
            self.CopyModelToCPU()
            self.t_op.current_step = step
            best_net = self.DeepCopy()
            best_t_op = CopyOperation(self.t_op)
        # Evaluate on test set.
        self.Evaluate(validation=False, collect_predictions=collect_predictions)
        sys.stdout.write('\n')
        if self.ShowNow(step):
          self.Show()
      if self.SaveNow(step):
        self.t_op.current_step = step
        self.CopyModelToCPU()
        util.WriteCheckpointFile(self.net, self.t_op)
        if dump_best:
          dump_best = False
          print 'Best error : %.4f' % best_error
          util.WriteCheckpointFile(best_net, best_t_op, best=True)

      stop = self.TrainStopCondition(step)
