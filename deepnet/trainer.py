from neuralnet import *
from fastdropoutnet import *
from dbm import *
from dbn import *
from sparse_coder import *
from choose_matrix_library import *
import numpy as np
from time import sleep

def LockGPU(max_retries=10):
  for retry_count in range(max_retries):
    board = gpu_lock.obtain_lock_id()
    if board != -1:
      break
    sleep(1)
  if board == -1:
    print 'No GPU board available.'
    sys.exit(1)
  else:
    cm.cuda_set_device(board)
    cm.cublas_init()
  return board

def FreeGPU(board):
  cm.cublas_shutdown()
  #gpu_lock.free_lock(board)

def LoadExperiment(model_file, train_op_file, eval_op_file):
  model = util.ReadModel(model_file)
  train_op = util.ReadOperation(train_op_file)
  eval_op = util.ReadOperation(eval_op_file)
  return model, train_op, eval_op

def CreateDeepnet(model, train_op, eval_op):
  if model.model_type == deepnet_pb2.Model.FEED_FORWARD_NET:
    return NeuralNet(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.DBM:
    return DBM(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.DBN:
    return DBN(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.SPARSE_CODER:
    return SparseCoder(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.FAST_DROPOUT_NET:
    return FastDropoutNet(model, train_op, eval_op)
  else:
    raise Exception('Model not implemented.')

def main():
  if use_gpu == 'yes':
    board = LockGPU()
  model, train_op, eval_op = LoadExperiment(sys.argv[1], sys.argv[2],
                                            sys.argv[3])
  model = CreateDeepnet(model, train_op, eval_op)
  model.Train()
  if use_gpu == 'yes':
    FreeGPU(board)
  #raw_input('Press Enter.')

if __name__ == '__main__':
  main()
