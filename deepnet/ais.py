"""Computes partition function for RBM-like models using Annealed Importance Sampling."""
import numpy as np
from deepnet import dbm
from deepnet import util
from deepnet import trainer as tr
from choose_matrix_library import *
import sys
import numpy as np
import pdb
import time
import itertools
import matplotlib.pyplot as plt
from deepnet import visualize
import lightspeed

def SampleEnergySoftmax(layer, numsamples, use_lightspeed=False):
  sample = layer.sample
  energy = layer.state
  temp = layer.expanded_batch
  if use_lightspeed:
    layer.ApplyActivation()
    layer.state.sum(axis=0, target=layer.temp)
    layer.state.div_by_row(layer.temp, target=temp)
    probs_cpu = temp.asarray().astype(np.float64)
    samples_cpu = lightspeed.SampleSoftmax(probs_cpu, numsamples)
    sample.overwrite(samples_cpu.astype(np.float32))
  else:
    sample.assign(0)
    for i in range(numsamples):
      energy.perturb_energy_for_softmax_sampling(target=temp)
      temp.choose_max_and_accumulate(sample)

def LogMeanExp(x):
  offset = x.max()
  return offset + np.log(np.exp(x-offset).mean())

def LogSumExp(x):
  offset = x.max()
  return offset + np.log(np.exp(x-offset).sum())

def Display(w, hid_state, input_state, w_var, x_axis):
  w = w.asarray().flatten()
  #plt.figure(1)
  #plt.clf()
  #plt.hist(w, 100)
  #visualize.display_hidden(hid_state.asarray(), 2, 'activations', prob=True)
  #plt.figure(3)
  #plt.clf()
  #plt.imshow(hid_state.asarray().T, cmap=plt.cm.gray, interpolation='nearest')
  #plt.figure(4)
  #plt.clf()
  #plt.imshow(input_state.asarray().T, cmap=plt.cm.gray, interpolation='nearest')
  #, state.shape[0], state.shape[1], state.shape[0], 3, title='Markov chains')
  #plt.tight_layout(pad=0, w_pad=0, h_pad=0)
  plt.figure(5)
  plt.clf()
  plt.suptitle('Variance')
  plt.plot(np.array(x_axis), np.array(w_var))
  plt.draw()

def AISReplicatedSoftmax(model, D, num_chains, display=False):
  schedule = np.concatenate((
    #np.arange(0.0, 1.0, 0.01),
    #np.arange(0.0, 1.0, 0.001),
    np.arange(0.0, 0.7, 0.001),  # 700
    np.arange(0.7, 0.9, 0.0001),  # 2000
    np.arange(0.9, 1.0, 0.00002)  # 5000
    ))
  #schedule = np.array([0.])
  cm.CUDAMatrix.init_random(seed=0)

  assert len(model.layer) == 2, 'Only implemented for RBMs.'
  steps = len(schedule)
  input_layer = model.layer[0]
  hidden_layer = model.layer[1]
  edge = model.edge[0]
  batchsize = num_chains
  w = edge.params['weight']
  a = hidden_layer.params['bias']
  b = input_layer.params['bias']
  numvis, numhid = w.shape
  f = 0.1
  input_layer.AllocateBatchsizeDependentMemory(num_chains)
  hidden_layer.AllocateBatchsizeDependentMemory(num_chains)

  # INITIALIZE TO SAMPLES FROM BASE MODEL.
  input_layer.state.assign(0)
  input_layer.NN.assign(D)
  input_layer.state.add_col_mult(b, f)
  SampleEnergySoftmax(input_layer, D)
  w_ais = cm.CUDAMatrix(np.zeros((1, batchsize)))
  #pdb.set_trace()

  w_variance = []
  x_axis = []
  if display:
    Display(w_ais, hidden_layer.state, input_layer.state, w_variance, x_axis)
    #raw_input('Press Enter.')
  #pdb.set_trace()

  # RUN AIS.
  for i in range(steps-1):
    sys.stdout.write('\r%d' % (i+1))
    sys.stdout.flush()
    cm.dot(w.T, input_layer.sample, target=hidden_layer.state)
    hidden_layer.state.add_col_mult(a, D)

    hidden_layer.state.mult(schedule[i], target=hidden_layer.temp)
    hidden_layer.state.mult(schedule[i+1])
    cm.log_1_plus_exp(hidden_layer.state, target=hidden_layer.deriv)
    cm.log_1_plus_exp(hidden_layer.temp)
    hidden_layer.deriv.subtract(hidden_layer.temp)
    w_ais.add_sums(hidden_layer.deriv, axis=0)
    w_ais.add_dot(b.T, input_layer.sample, mult=(1-f)*(schedule[i+1]-schedule[i]))

    hidden_layer.ApplyActivation()
    hidden_layer.Sample()
    cm.dot(w, hidden_layer.sample, target=input_layer.state)
    input_layer.state.add_col_vec(b)
    input_layer.state.mult(schedule[i+1])
    input_layer.state.add_col_mult(b, f*(1-schedule[i+1]))
    SampleEnergySoftmax(input_layer, D)
    if display and (i % 100 == 0 or i == steps - 2):
      w_variance.append(w_ais.asarray().var())
      x_axis.append(i)
      Display(w_ais, hidden_layer.state, input_layer.sample, w_variance, x_axis)
  sys.stdout.write('\n')
  z = LogMeanExp(w_ais.asarray()) + D * LogSumExp(f * b.asarray()) + numhid * np.log(2)
  return z

def AISBinaryRbm(model, schedule):
  cm.CUDAMatrix.init_random(seed=int(time.time()))
  assert len(model.layer) == 2, 'Only implemented for RBMs.'
  steps = len(schedule)
  input_layer = model.layer[0]
  hidden_layer = model.layer[1]
  edge = model.edge[0]
  batchsize = model.t_op.batchsize
  w = edge.params['weight']
  a = hidden_layer.params['bias']
  b = input_layer.params['bias']
  numvis, numhid = w.shape

  # INITIALIZE TO UNIFORM RANDOM.
  input_layer.state.assign(0)
  input_layer.ApplyActivation()
  input_layer.Sample()
  w_ais = cm.CUDAMatrix(np.zeros((1, batchsize)))
  unitcell = cm.empty((1, 1))
  # RUN AIS.
  for i in range(1, steps):
    cm.dot(w.T, input_layer.sample, target=hidden_layer.state)
    hidden_layer.state.add_col_vec(a)

    hidden_layer.state.mult(schedule[i-1], target=hidden_layer.temp)
    hidden_layer.state.mult(schedule[i])
    cm.log_1_plus_exp(hidden_layer.state, target=hidden_layer.deriv)
    cm.log_1_plus_exp(hidden_layer.temp)
    hidden_layer.deriv.subtract(hidden_layer.temp)
    w_ais.add_sums(hidden_layer.deriv, axis=0)
    w_ais.add_dot(b.T, input_layer.state, mult=schedule[i]-schedule[i-1])

    hidden_layer.ApplyActivation()
    hidden_layer.Sample()
    cm.dot(w, hidden_layer.sample, target=input_layer.state)
    input_layer.state.add_col_vec(b)
    input_layer.state.mult(schedule[i])
    input_layer.ApplyActivation()
    input_layer.Sample()
  z = LogMeanExp(w_ais.asarray()) + numvis * np.log(2) + numhid * np.log(2)
  return z

def GetAll(n):
  x = np.zeros((n, 2**n))
  a = []
  for i in range(n):
    a.append([0, 1])
  for i, r in enumerate(itertools.product(*tuple(a))):
    x[:, i] = np.array(r)
  return x

def ExactZ_binary_binary(model):
  assert len(model.layer) == 2, 'Only implemented for RBMs.'
  steps = len(schedule)
  input_layer = model.layer[0]
  hidden_layer = model.layer[1]
  edge = model.edge[0]
  w = edge.params['weight']
  a = hidden_layer.params['bias']
  b = input_layer.params['bias']
  numvis, numhid = w.shape
  batchsize = 2**numvis
  input_layer.AllocateBatchsizeDependentMemory(batchsize)
  hidden_layer.AllocateBatchsizeDependentMemory(batchsize)
  all_inputs = GetAll(numvis)
  w_ais = cm.CUDAMatrix(np.zeros((1, batchsize)))
  input_layer.sample.overwrite(all_inputs)
  cm.dot(w.T, input_layer.sample, target=hidden_layer.state)
  hidden_layer.state.add_col_vec(a)
  cm.log_1_plus_exp(hidden_layer.state)
  w_ais.add_sums(hidden_layer.state, axis=0)
  w_ais.add_dot(b.T, input_layer.state)
  offset = float(w_ais.asarray().max())
  w_ais.subtract(offset)
  cm.exp(w_ais)
  z = offset + np.log(w_ais.asarray().sum())
  return z

def Usage():
  print '%s <model file> <number of Markov chains to run> [number of words (for Replicated Softmax models)]'

if __name__ == '__main__':
  board = tr.LockGPU()
  model_file = sys.argv[1]
  numchains = int(sys.argv[2])
  if len(sys.argv) > 3:
    D = int(sys.argv[3]) #10 # number of words.
  m = dbm.DBM(model_file)
  m.LoadModelOnGPU(batchsize=numchains)
  plt.ion()
  log_z = AISReplicatedSoftmax(m, D, numchains, display=True)
  print 'Log Z %.5f' % log_z
  #log_z = AIS(m, schedule)
  #print 'Log Z %.5f' % log_z
  #log_z = ExactZ_binary_binary(m)
  #print 'Exact %.5f' % log_z
  tr.FreeGPU(board)
  raw_input('Press Enter.')
