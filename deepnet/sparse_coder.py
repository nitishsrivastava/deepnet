from neuralnet import *
from sparse_code_layer import *
import scipy.linalg

class SparseCoder(NeuralNet):
  def SetLayerAndEdgeClass(self):
    self.LayerClass = SparseCodeLayer
    self.EdgeClass = Edge

  def Show(self):
    #encoder = self.encoder.params['weight'].asarray()
    decoder = self.decoder.params['weight'].asarray()
    recon = self.input_layer.approximator.asarray()
    #visualize.display_wsorted(encoder, 28, 16, 16, 1, title='encoder')
    visualize.display_wsorted(decoder.T, 28, 16, 16, 2, title='decoder')
    visualize.display_w(recon[:, :100], 28, 10, 10, 3, title='reconstructions')

  def Sort(self):
    assert len(self.layer) == 2
    assert len(self.edge) == 2
    if self.layer[0].is_input:
      self.input_layer = self.layer[0]
      self.code_layer = self.layer[1]
    else:
      self.input_layer = self.layer[1]
      self.code_layer = self.layer[0]
    if self.edge[0].node1 == self.input_layer:
      self.encoder = self.edge[0]
      self.decoder = self.edge[1]
    else:
      self.encoder = self.edge[1]
      self.decoder = self.edge[0]
    return [self.input_layer, self.code_layer]

  @staticmethod
  def SolveForZ(x, z_est, wd, alpha, beta, temp, temp2, z):
    """Solve for z in (alpha + wd.wd^T)z = wd . x + alpha z_est - beta.

    Output goes in z. temp is a matrix to store wd^Twd.
    """
    cm.dot(wd, wd.T, target=temp)
    temp.add(alpha)
    z_est.mult(alpha, target=temp2)
    temp2.add_dot(wd, x)
    temp2.subtract(beta)

    # Copy matrices to cpu.
    A = temp.asarray()
    B = temp2.asarray()

    # Solve AZ = B
    Z = scipy.linalg.solve(A, B, overwrite_a=True, overwrite_b=True)
    
    # Copy result back to gpu.
    z.overwrite(Z)


  def ForwardPropagate(self, train=False):
    """Loads input and computes the sparse code for it."""

    input_layer = self.input_layer
    code_layer = self.code_layer

    # Load data into state.
    input_layer.GetData()

    # Run it through the encoder.
    inputs = input_layer.state
    we = self.encoder.params['weight']
    be = code_layer.params['bias']
    code_approx = code_layer.approximator
    cm.dot(we.T, inputs, target=code_approx)
    code_approx.add_col_vec(be)
    code_layer.ApplyActivation(code_approx)

    # Solve for optimal z.
    z = code_layer.state
    wd = self.decoder.params['weight']
    hyp = code_layer.hyperparams
    alpha = hyp.sc_alpha
    beta = hyp.sc_beta
    temp = code_layer.m_by_m
    temp2 = code_layer.deriv
    SparseCoder.SolveForZ(inputs, code_approx, wd, alpha, beta, temp, temp2, z)

    if hyp.dropout:
      if train:
        # Randomly set states to zero.
        code_layer.mask.fill_with_rand()
        code_layer.mask.greater_than(hyp.dropout_prob)
        z.mult(code_layer.mask)
      else:
        # Produce expected output.
        z.mult(1.0 - hyp.dropout_prob)



  def GetLoss(self):
    """Computes loss and its derivatives."""

    input_layer = self.input_layer
    code_layer = self.code_layer

    # Decode z.
    wd = self.decoder.params['weight']
    bd = input_layer.params['bias']
    z = code_layer.state
    input_recon = input_layer.approximator
    cm.dot(wd.T, z, target=input_recon)
    input_recon.add_col_vec(bd)

    # Compute loss function.
    code_approx = code_layer.approximator
    hyp = code_layer.hyperparams
    alpha = hyp.sc_alpha
    beta = hyp.sc_beta

    input_recon.subtract(input_layer.state, target=input_layer.deriv)  # input reconstruction residual.
    code_approx.subtract(z, target=code_layer.deriv)  # code construction residual.
    cm.abs(z, target=code_layer.temp)  # L1 norm of code.
    code_layer.temp.sum(axis=1, target=code_layer.dimsize)
    code_layer.dimsize.sum(axis=0, target=code_layer.unitcell)
    loss = input_layer.deriv.euclid_norm()**2 +\
        alpha * code_layer.deriv.euclid_norm()**2 +\
        beta * code_layer.unitcell.euclid_norm()
    perf = deepnet_pb2.Metrics()
    perf.MergeFrom(code_layer.proto.performance_stats)
    perf.count = self.batchsize
    perf.error = loss

    return perf

  def UpdateParameters(self, step):
    """Update the encoder and decoder weigths and biases.
    Args:
      step: Time step of training.
    """
    numcases = self.batchsize
    code_layer = self.code_layer
    input_layer = self.input_layer
    wd = self.decoder.params['weight']
    bd = input_layer.params['bias']
    z = code_layer.state
    inputs = input_layer.state
    we = self.encoder.params['weight']
    be = code_layer.params['bias']
    code_approx = code_layer.approximator
    hyp = code_layer.hyperparams
    alpha = hyp.sc_alpha
    beta = hyp.sc_beta

    # Derivatives for decoder weights.
    deriv = input_layer.deriv
    momentum, epsilon = self.decoder.GetMomentumAndEpsilon(step)
    wd_delta = self.decoder.grad_weight
    wd_delta.mult(momentum)
    wd_delta.add_dot(z, deriv.T, 1.0 / numcases)
    wd.add_mult(wd_delta, -epsilon)

    # Derivatives for decoder bias.
    momentum, epsilon = input_layer.GetMomentumAndEpsilon(step)
    bd_delta = input_layer.grad_bias
    bd_delta.mult(momentum)
    bd_delta.add_sums(deriv, axis=1, mult=1.0 / numcases)
    bd.add_mult(bd_delta, -epsilon)

    # Derivatives for encoder weights.
    code_layer.ComputeDeriv(code_approx)  # backprop through non-linearity.
    deriv = code_layer.deriv
    momentum, epsilon = self.encoder.GetMomentumAndEpsilon(step)
    we_delta = self.encoder.grad_weight
    we_delta.mult(momentum)
    we_delta.add_dot(inputs, deriv.T, alpha / numcases)
    we.add_mult(we_delta, -epsilon)

    # Derivatives for encoder bias.
    momentum, epsilon = code_layer.GetMomentumAndEpsilon(step)
    be_delta = code_layer.grad_bias
    be_delta.mult(momentum)
    be_delta.add_sums(deriv, axis=1, mult=1.0 / numcases)
    be.add_mult(be_delta, -epsilon)

  def EvaluateOneBatch(self):
    """Evaluate on one mini-batch.
    Args:
      step: Training step.
    """
    self.ForwardPropagate()
    return [self.GetLoss()]

  def TrainOneBatch(self, step):
    """Train using one mini-batch.
    Args:
      step: Training step.
    """
    """
    if step > self.code_layer.hyperparams.switch_on_sc_alpha_after:
      self.code_layer.hyperparams.sc_alpha = 1.0
    """
    self.ForwardPropagate(train=True)
    losses = [self.GetLoss()]
    self.UpdateParameters(step)
    return losses
