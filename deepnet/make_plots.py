from deepnet import deepnet_pb2
import matplotlib.pyplot as plt
import glob, sys, gzip, numpy as np

def preds(metrics_list):
  y = []
  for metric in metrics_list:
    count = metric.count
    y.append( 100*(1- metric.correct_preds/metric.count))
  return y


def get_plot(v, skip, label):
  y = v[skip:]
  x = np.arange(skip, len(v))
  return plt.plot(x, y, label=label)


if __name__ == '__main__':
  plt.ion()
  proto = sys.argv[1]
  proto = glob.glob(proto + "*")[-1]
  print proto
  skip = 0
  if len(sys.argv) > 2:
    skip = int(sys.argv[2])
  model_pb = deepnet_pb2.Model()
  f = gzip.open(proto, 'rb')
  model_pb.ParseFromString(f.read())
  f.close()
  train = preds(model_pb.train_stats)
  valid = preds(model_pb.validation_stats)
  test = preds(model_pb.test_stats)
  x = np.arange(len(train))
  plt.figure(1)
  p1 = get_plot(train, skip, 'train')
  p2 = get_plot(valid, skip, 'valid')
  p3 = get_plot(test, skip, 'test')
  plt.legend()
  plt.xlabel('Iterations / 2000')
  plt.ylabel('Error %')
  plt.draw()
  raw_input('Press any key')
