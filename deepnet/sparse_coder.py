from neuralnet import *
from sparse_code_layer import *
import scipy.linalg

class SparseCoder(NeuralNet):
  def SetLayerAndEdgeClass(self):
    self.LayerClass = SparseCodeLayer
    self.EdgeClass = Edge

  def Show(self):
    encoder = self.encoder.params['weight'].asarray()
    decoder = self.decoder.params['weight'].asarray()
    recon = self.input_layer.approximator.asarray()
    recep_field = self.encoder.proto.receptive_field_width
    rows = self.encoder.proto.display_rows
    cols = self.encoder.proto.display_cols
    visualize.display_wsorted(encoder, recep_field, rows, cols, 1, title='encoder')
    visualize.display_wsorted(decoder.T, recep_field, rows, cols, 2, title='decoder')
    visualize.display_w(recon[:, :100], recep_field, 10, 10, 3, title='reconstructions')
    visualize.display_hidden(self.code_layer.state.asarray(), 4, 'code distribution', prob=False)

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

  def SolveForZ(self):
    """Solve for z in (alpha + beta.wd.wd^T)z = wd . x + alpha z_est - gamma exactly.

    Output goes in z. temp is a matrix to store wd^Twd.
    """
    input_layer = self.input_layer
    code_layer = self.code_layer
    z = code_layer.state
    wd = self.decoder.params['weight']
    hyp = code_layer.hyperparams
    alpha = hyp.sc_alpha
    beta = hyp.sc_beta
    gamma = hyp.sc_gamma
    temp = code_layer.m_by_m
    temp2 = code_layer.deriv
    eye_m_by_m = code_layer.eye_m_by_m

    cm.dot(wd, wd.T, target=temp)
    temp.mult(beta)
    temp.add(alpha)
    z_est.mult(alpha, target=temp2)
    temp2.add_dot(wd, x, mult=beta)
    temp2.subtract(gamma)

    # Copy matrices to cpu.
    A = temp.asarray()
    B = temp2.asarray()

    # Solve AZ = B
    Z = scipy.linalg.solve(A, B, overwrite_a=True, overwrite_b=True)
    
    # Copy result back to gpu.
    z.overwrite(Z)

  def IterateForZ(self, train=False):
    """Solve for z in (alpha + beta.wd.wd^T)z = wd . x + alpha z_est - gamma using gradient descent.

    Output goes in z. temp is a matrix to store wd^Twd.
    """
    input_layer = self.input_layer
    code_layer = self.code_layer
    epsilon = 0.01
    steps = 20
    z = code_layer.state
    wd = self.decoder.params['weight']
    hyp = code_layer.hyperparams
    alpha = hyp.sc_alpha
    beta = hyp.sc_beta
    gamma = hyp.sc_gamma
    temp = code_layer.m_by_m
    temp2 = code_layer.deriv
    temp3 = code_layer.temp3  # This is bad! use better names.
    grad = code_layer.grad
    z_est = code_layer.approximator

    avg_models = hyp.dropout and (not hyp.dropout or not train)

    cm.dot(wd, wd.T, target=temp)
    temp.mult(beta)

    if avg_models:
      temp.mult((1.0 - hyp.dropout_prob)**2)
      temp.mult_diagonal(1. / (1.0 - hyp.dropout_prob))

    temp.add_diagonal(alpha)

    z_est.mult(alpha, target=temp2)

    if avg_models:
      temp2.add_dot(wd, input_layer.state, mult=beta * (1.0 - hyp.dropout_prob))
      #temp2.add_dot(wd, input_layer.state, mult=beta)
    elif hyp.dropout:
      temp2.add_dot(wd, input_layer.state, mult=beta)
      temp2.mult(code_layer.mask)
    else:
      temp2.add_dot(wd, input_layer.state, mult=beta)
    z.assign(z_est)

    #pdb.set_trace()
    for i in range(steps):
      cm.dot(temp, z, target=grad)
      grad.subtract(temp2)
      z.sign(target=temp3)
      grad.add_mult(temp3, alpha=gamma)
      if hyp.dropout and train:
        #code_layer.mask.fill_with_rand()
        #code_layer.mask.greater_than(hyp.dropout_prob)
        grad.mult(code_layer.mask)
      z.add_mult(grad, alpha=-epsilon)
    #pdb.set_trace()


  def ForwardPropagate(self, train=False, method='iter'):
    """Loads input and computes the sparse code for it."""

    input_layer = self.input_layer
    code_layer = self.code_layer

    # Load data into state.
    input_layer.GetData()

    # Run it through the encoder.
    inputs = input_layer.state
    we = self.encoder.params['weight']
    be = code_layer.params['bias']
    scale = code_layer.params['scale']
    hyp = code_layer.hyperparams
    code_approx = code_layer.approximator
    cm.dot(we.T, inputs, target=code_approx)
    code_approx.add_col_vec(be)
    code_layer.ApplyActivation(code_approx)
    code_approx.mult_by_col(scale)
    if hyp.dropout and train:
      code_layer.mask.fill_with_rand()
      code_layer.mask.greater_than(hyp.dropout_prob)
      code_approx.mult(code_layer.mask)

    # Infer z.
    if train:
      if method == 'iter':
        self.IterateForZ(train=train)
      elif method == 'exact':
        self.SolveForZ()
    else:
      if method == 'iter':
        self.IterateForZ(train=train)
      #code_layer.state.assign(code_approx)

  def GetLoss(self, train=False):
    """Computes loss and its derivatives."""

    input_layer = self.input_layer
    code_layer = self.code_layer

    # Decode z.
    hyp = code_layer.hyperparams
    wd = self.decoder.params['weight']
    bd = input_layer.params['bias']
    z = code_layer.state
    input_recon = input_layer.approximator
    cm.dot(wd.T, z, target=input_recon)
    input_recon.add_col_vec(bd)

    # Compute loss function.
    code_approx = code_layer.approximator
    alpha = hyp.sc_alpha
    gamma = hyp.sc_gamma
    beta = hyp.sc_beta
    input_recon.subtract(input_layer.state, target=input_layer.deriv)  # input reconstruction residual.
    code_approx.subtract(z, target=code_layer.deriv)  # code construction residual.
    cm.abs(z, target=code_layer.temp)  # L1 norm of code.
    code_layer.temp.sum(axis=1, target=code_layer.dimsize)
    code_layer.dimsize.sum(axis=0, target=code_layer.unitcell)
    loss1 = 0.5 * beta * input_layer.deriv.euclid_norm()**2
    loss2 = 0.5 * alpha * code_layer.deriv.euclid_norm()**2
    loss3 = gamma * code_layer.unitcell.euclid_norm()
    loss4 = loss1 + loss2 + loss3
    err = []
    for l in [loss1, loss2, loss3, loss4]:
      perf = deepnet_pb2.Metrics()
      perf.MergeFrom(code_layer.proto.performance_stats)
      perf.count = self.batchsize
      perf.error = l
      err.append(perf)
    return err

  def UpdateParameters(self, step):
    """Update the encoder and decoder weigths and biases.
    Args:
      step: Time step of training.
    """
    numcases = self.batchsize
    code_layer = self.code_layer
    input_layer = self.input_layer
    encoder = self.encoder
    decoder = self.decoder
    wd = decoder.params['weight']
    bd = input_layer.params['bias']
    z = code_layer.state
    inputs = input_layer.state
    we = encoder.params['weight']
    be = code_layer.params['bias']
    scale = code_layer.params['scale']
    code_approx = code_layer.approximator
    hyp = code_layer.hyperparams
    alpha = hyp.sc_alpha
    beta = hyp.sc_beta
    gamma = hyp.sc_gamma
    enc_hyp = encoder.hyperparams
    dec_hyp = decoder.hyperparams

    # Derivatives for decoder weights.
    deriv = input_layer.deriv
    momentum, epsilon = decoder.GetMomentumAndEpsilon(step)
    wd_delta = self.decoder.grad_weight
    wd_delta.mult(momentum)
    wd_delta.add_dot(z, deriv.T, beta / numcases)
    if dec_hyp.apply_l2_decay:
      wd_delta.add_mult(wd, alpha=dec_hyp.l2_decay)

    # Derivatives for decoder bias.
    momentum, epsilon = input_layer.GetMomentumAndEpsilon(step)
    bd_delta = input_layer.grad_bias
    bd_delta.mult(momentum)
    bd_delta.add_sums(deriv, axis=1, mult=beta / numcases)

    # Derivatives for scale.
    deriv = code_layer.deriv
    code_approx.div_by_col(scale)
    scale_delta = code_layer.grad_scale
    scale_delta.mult(momentum)
    temp = code_layer.temp3
    code_approx.mult(deriv, target=temp)
    scale_delta.add_sums(temp, axis=1, mult=alpha / numcases)

    # Derivatives for encoder weights.
    code_layer.deriv.mult_by_col(scale)
    code_layer.ComputeDeriv(code_approx)  # backprop through non-linearity.
    deriv = code_layer.deriv
    momentum, epsilon = encoder.GetMomentumAndEpsilon(step)
    we_delta = self.encoder.grad_weight
    we_delta.mult(momentum)
    we_delta.add_dot(inputs, deriv.T, alpha / numcases)
    if enc_hyp.apply_l2_decay:
      we_delta.add_mult(we, alpha=enc_hyp.l2_decay)

    # Derivatives for encoder bias.
    momentum, epsilon = code_layer.GetMomentumAndEpsilon(step)
    be_delta = code_layer.grad_bias
    be_delta.mult(momentum)
    be_delta.add_sums(deriv, axis=1, mult=alpha / numcases)

    # Apply the updates.
    scale.add_mult(scale_delta, -epsilon)
    bd.add_mult(bd_delta, -epsilon)
    wd.add_mult(wd_delta, -epsilon)
    be.add_mult(be_delta, -epsilon)
    we.add_mult(we_delta, -epsilon)

    if dec_hyp.apply_weight_norm:
      wd.norm_limit(dec_hyp.weight_norm, axis=0)
    if enc_hyp.apply_weight_norm:
      we.norm_limit(enc_hyp.weight_norm, axis=0)

  def EvaluateOneBatch(self):
    """Evaluate on one mini-batch.
    Args:
      step: Training step.
    """
    self.ForwardPropagate()
    return self.GetLoss()

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
    losses = self.GetLoss(train=True)
    self.UpdateParameters(step)
    return losses
