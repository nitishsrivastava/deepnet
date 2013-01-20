"""Monte Carlo model averaging for dropout networks."""
from neuralnet import *
from trainer import *
import glob
import sys
import random

def ExtractRepresentations(model_file, train_op_file, layernames,
                           base_output_dir, memory = '100M', k=10):
  LockGPU()
  model = util.ReadModel(model_file)
  op = ReadOperation(train_op_file)
  op.randomize = False
  net = CreateDeepnet(model, op, op)
  net.LoadModelOnGPU()
  net.SetUpData()
  for i in range(k):
    output_dir = os.path.join(base_output_dir, 'sample_%.5d' % i) 
    sys.stdout.write('\r Sample %d' % (i+1))
    sys.stdout.flush()
    net.WriteRepresentationToDisk(layernames, output_dir, memory=memory, drop=True)
  sys.stdout.write('\n')
  FreeGPU()


def GetAverageResult(truth_file, pred_dir, total, k, avg_over=10):
  sample_ids = range(total)
  x = []
  pred_dict = {}
  truth = np.load(truth_file)
  for t in range(avg_over):
    avg_pred = None
    for j in range(k):
      i = random.choice(sample_ids)
      prediction_file = glob.glob(os.path.join(pred_dir, 'sample_%.5d' % i, '*.npy'))[0]
      predictions = pred_dict.get(i, np.load(prediction_file))
      pred_dict[i] = predictions 
      if avg_pred is None:
        avg_pred = predictions
      else:
        avg_pred += predictions
    avg_pred /= k
    pred = avg_pred.argmax(axis=1)
    error = len((pred - truth).nonzero()[0])
    x.append((100. * error) / len(truth))
  x = np.array(x)
  return x.mean(), x.std()

def main():
  model_file = sys.argv[1]
  model = util.ReadModel(model_file)
  train_op_file = sys.argv[2]
  output_dir = sys.argv[3]
  layernames = ['output_layer']
  total = 1000
  k = 200
  avg_over = 100

  true_label_file = '/ais/gobi3/u/nitish/mnist/test_labels.npy'
  plot_data_file = '/ais/gobi3/u/nitish/mnist/results/mc_avg.npy'
  #ExtractRepresentations(model_file, train_op_file, layernames, output_dir, memory='1G', k=total)
  out = np.zeros((k, 3))
  for l in range(1, k+1):
    mean, std = GetAverageResult(true_label_file, output_dir, total, l, avg_over=avg_over)
    print '%d %.4f %.4f' % (l, mean, std)
    out[l-1, 0] = l
    out[l-1, 1] = mean
    out[l-1, 2] = std
  np.save(plot_data_file, out)


if __name__ == '__main__':
  main()
