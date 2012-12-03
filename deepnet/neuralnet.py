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
    s += ' %s_E: %.3f' % (prefix, stat.error / stat.count)
  if stat.compute_MAP:
    s += ' %s_MAP: %.3f' % (prefix, stat.MAP)
  if stat.compute_prec50:
    s += ' %s_prec50: %.3f' % (prefix, stat.prec50)
  if stat.compute_sparsity:
    s += ' %s_sp: %.3f' % (prefix, stat.sparsity)
  return s

def Accumulate(acc, perf):
 acc.count += perf.count
 acc.cross_entropy += perf.cross_entropy
 acc.error += perf.error
 acc.correct_preds += perf.correct_preds

class NeuralNet(object):
  def __init__(self, net, t_op, e_op):
    if isinstance(net, deepnet_pb2.Model):
      self.net = net
    else:
      self.net = ReadModel(net)
    if isinstance(t_op, deepnet_pb2.Operation):
      self.t_op = t_op
    else:
      self.t_op = ReadOperation(t_op)
    if isinstance(e_op, deepnet_pb2.Operation):
      self.e_op = e_op
    else:
      self.e_op = ReadOperation(e_op)
    cm.CUDAMatrix.init_random(self.net.seed)
    np.random.seed(self.net.seed)
    self.data = None
    self.layer = []
    self.input_datalayer = []
    self.output_datalayer = []
    self.edge = []
    self.unclamped_layer = []
    self.SetLayerAndEdgeClass()

  def SetLayerAndEdgeClass(self):
    self.LayerClass = Layer
    self.EdgeClass = Edge

  def PrintNetwork(self):
    for layer in self.layer:
      print layer.name
      layer.PrintNeighbours()

  def LoadModelOnGPU(self, batchsize=-1):
    """Load the model on the GPU."""
    if batchsize < 0:
      batchsize=self.t_op.batchsize

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
    for node in self.node_list:
      node.ComputeUp(train, step, self.train_stop_steps)

  def BackwardPropagate(self, step):
    """Backprop through the network.

    Args:
      step: Training step.
    """
    loss_list = []
    for node in reversed(self.node_list):
      loss = node.ComputeDown(step)
      if loss:
        loss_list.append(loss)
    return loss_list

  def TrainOneBatch(self, step):
    """Train once on one mini-batch.

    Args:
      step: Training step.
    Returns:
      List of losses incurred at each output layer.
    """
    self.ForwardPropagate(train=True)
    losses = self.BackwardPropagate(step)
    return losses

  def EvaluateOneBatch(self):
    """Evaluate one mini-batch."""
    self.ForwardPropagate()
    return [node.GetLoss() for node in self.output_datalayer]

  def Evaluate(self, validation=True):
    """Evaluate the model.
    Args:
      validation: If True, evaluate on the validation set,
        else evaluate on test set.
    """
    step = 0
    stats = []
    if validation:
      datagetter = self.GetValidationBatch
      stopcondition = self.ValidationStopCondition
      prefix = 'V'
      stats_list = self.net.validation_stats
    else:
      datagetter = self.GetTestBatch
      stopcondition = self.TestStopCondition
      prefix = 'E'
      stats_list = self.net.test_stats
    stop = stopcondition(0)
    while not stop:
      datagetter()
      losses = self.EvaluateOneBatch()
      if stats:
        for loss, acc in zip(losses, stats):
          Accumulate(acc, loss)
      else:
        stats = losses
      step += 1
      stop = stopcondition(step)
    for stat in stats:
      sys.stdout.write(GetPerformanceStats(stat, prefix=prefix))
      stats_list.extend(stats)

  def ScoreOneLabel(self, preds, targets):
    """Computes Average precision and precision at 50."""
    targets_sorted = targets[(-preds.T).argsort().flatten(),:]
    prec = targets_sorted.cumsum() / np.arange(1.0, 1 + targets.shape[0])
    recall = targets_sorted.cumsum() / float(sum(targets))
    ap = np.dot(prec, targets_sorted) / sum(targets)
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

  def Save(self):
    for layer in self.layer:
      layer.SaveParameters()
    for edge in self.edge:
      edge.SaveParameters()
    util.WriteCheckpointFile(self)

  def GetTrainBatch(self):
    for layer in self.input_datalayer:
      layer.GetTrainData()
    for layer in self.output_datalayer:
      layer.GetTrainData()

  def GetValidationBatch(self):
    for layer in self.input_datalayer:
      layer.GetValidationData()
    for layer in self.output_datalayer:
      layer.GetValidationData()

  def GetTestBatch(self):
    for layer in self.input_datalayer:
      layer.GetTestData()
    for layer in self.output_datalayer:
      layer.GetTestData()

  def SetUpData(self, train_link=None, valid_link=None, test_link=None):
    """Setup the data.

    Sets up data handlers. The optional links are other previously created data
    handlers that must be tied to the new data handlers.
    """
    data = util.ReadData(self.t_op.data_proto)
    for node in self.layer:
      if not node.proto.HasField('data_field'):
        continue
      data_field = node.proto.data_field
      if data_field.tied:
        layer_tied_to = next(l for l in self.layer\
                             if l.name == data_field.tied_to)
        print 'Layer %s is tied to %s' % (node.name, layer_tied_to.name)
        node.SetDataHandles(tied_to=layer_tied_to)
      else:
        if data_field.train:
          train_data = DataHandle(
            data_field.train,
            next(d for d in data.data if d.name == data_field.train),
            self.t_op, node.hyperparams, permutation_link=train_link)
        else:
          train_data = None
        if data_field.validation:
          validation_data = DataHandle(
            data_field.validation,
            next(d for d in data.data if d.name == data_field.validation),
            self.e_op, node.hyperparams, permutation_link=valid_link)
        else:
          validation_data = None
        if data_field.test:
          test_data = DataHandle(
            data_field.test,
            next(d for d in data.data if d.name == data_field.test),
            self.e_op, node.hyperparams, permutation_link=test_link)
        else:
          test_data = None
        if train_link is None:
          train_link = train_data
        if valid_link is None:
          valid_link = validation_data
        if test_link is None:
          test_link = test_data
        node.SetDataHandles(train=train_data, valid=validation_data,
                            test=test_data)
    return train_link, valid_link, test_link


  def SetUpTrainer(self, *args):
    """Load the model, setup the data, set the stopping conditions."""
    self.LoadModelOnGPU()
    #self.PrintNetwork()
    args = self.SetUpData(*args)
    data_layer = self.input_datalayer[0]
    if self.t_op.stopcondition.all_processed:
      num_steps = data_layer.train_data_handler.num_batches
    else:
      num_steps = self.t_op.stopcondition.steps
    self.train_stop_steps = num_steps
    if self.e_op.stopcondition.all_processed and data_layer.validation_data_handler:
      num_steps = data_layer.validation_data_handler.num_batches
    else:
      num_steps = self.e_op.stopcondition.steps
    self.validation_stop_steps = num_steps
    if self.e_op.stopcondition.all_processed and data_layer.test_data_handler:
      num_steps = data_layer.test_data_handler.num_batches
    else:
      num_steps = self.e_op.stopcondition.steps
    self.test_stop_steps = num_steps

    self.eval_now_steps = self.t_op.eval_after
    self.save_now_steps = self.t_op.checkpoint_after
    self.show_now_steps = self.t_op.show_after
    return args

  def Show(self):
    """Visualize the state of the layers and edges in the network."""
    for layer in self.layer:
      layer.Show()
    for edge in self.edge:
      edge.Show()

  def Train(self):
    """Train the model."""
    self.SetUpTrainer()
    step = self.t_op.current_step
    stop = self.TrainStopCondition(step)
    stats = []
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
        # Evaluation on validation set.
        self.Evaluate(validation=True)
        # Evaluation on test set.
        self.Evaluate(validation=False)
        sys.stdout.write('\n')
      if self.SaveNow(step):
        self.t_op.current_step = step
        self.Save()
      stop = self.TrainStopCondition(step)
