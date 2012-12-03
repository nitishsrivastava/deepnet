from neuralnet import *
from dbm import *
from dbn import *
import cudamat as cm
import numpy as np
from cudamat import gpu_lock
from time import sleep

def LockGPU(max_retries=10):
  for retry_count in range(max_retries):
    board = gpu_lock.obtain_lock_id()
    if board != -1:
      break
  if board == -1:
    print 'No GPU board available.'
    sys.exit(1)
  else:
    cm.cuda_set_device(board)
    cm.cublas_init()

def FreeGPU():
  cm.cublas_shutdown()

if __name__ == '__main__':
  LockGPU()
  model = util.ReadModel(sys.argv[1])
  train_op = util.ReadOperation(sys.argv[2])
  eval_op = util.ReadOperation(sys.argv[3])
  if model.model_type == deepnet_pb2.Model.FEED_FORWARD_NET:
    model = NeuralNet(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.DBM:
    model = DBM(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.DBN:
    model = DBN(model, train_op, eval_op)
  else:
    raise Exception('Model not implemented.')
  model.Train()
  FreeGPU()
