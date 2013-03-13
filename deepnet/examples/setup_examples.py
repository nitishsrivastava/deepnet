"""Sets the data path and output paths in trainers."""
from deepnet import util
from google.protobuf import text_format
import os.path
import sys

def MakeDataPbtxt(data_pbtxt_file, data_path):
  data_pbtxt = util.ReadData('mnist.pbtxt')
  for data in data_pbtxt.data:
    fname = os.path.basename(data.file_pattern)
    data.file_pattern = os.path.join(data_path, fname)
  util.WritePbtxt(data_pbtxt_file, data_pbtxt)

def MakeTrainers(trainer_file, data_pbtxt_file, output_path):
  trainer = util.ReadOperation(trainer_file)
  trainer.data_proto = data_pbtxt_file 
  trainer.checkpoint_directory = output_path
  util.WritePbtxt(trainer_file, trainer)

def EditPretrainedModels(p, output_path):
  pm = []
  for m in p.pretrained_model:
    fname = os.path.basename(m)
    pm.append(os.path.join(output_path, fname))
  if pm:
    del p.pretrained_model[:]
    p.pretrained_model.extend(pm)

def MakeModels(model_file, output_path):
  model = util.ReadModel(model_file)
  for l in model.layer:
    for p in l.param:
      EditPretrainedModels(p, output_path)
  for e in model.edge:
    for p in e.param:
      EditPretrainedModels(p, output_path)
  util.WritePbtxt(model_file, model)

def main():
  data_path = os.path.abspath(sys.argv[1])  # Path to mnist data directory.
  output_path = os.path.abspath(sys.argv[2])  # Path where learned models will be written.

  data_pbtxt_file = os.path.join(data_path, 'mnist.pbtxt')
  MakeDataPbtxt(data_pbtxt_file, data_path)

  for model in ['ae', 'dbm', 'dbn', 'ff', 'rbm', 'convnet']:
    trainer_file = os.path.join(model, 'train.pbtxt')
    MakeTrainers(trainer_file, data_pbtxt_file, output_path)

  for model in ['dbn']:
    trainer_file = os.path.join(model, 'train_classifier.pbtxt')
    MakeTrainers(trainer_file, data_pbtxt_file, output_path)

  model_files = [
    os.path.join('ae', 'model_layer1.pbtxt'),
    os.path.join('ae', 'model_layer2.pbtxt'),
    os.path.join('ae', 'classifier.pbtxt'),
    os.path.join('dbm', 'model.pbtxt'),
    os.path.join('dbn', 'mnist_rbm1.pbtxt'),
    os.path.join('dbn', 'mnist_rbm2.pbtxt'),
    os.path.join('dbn', 'mnist_rbm3.pbtxt'),
    os.path.join('dbn', 'mnist_classifier.pbtxt'),
    os.path.join('ff', 'model.pbtxt'),
    os.path.join('ff', 'model_dropout.pbtxt'),
    os.path.join('rbm', 'model.pbtxt'),
    os.path.join('convnet', 'model_conv.pbtxt'),
  ]
  for model_file in model_files:
    MakeModels(model_file, output_path)

if __name__ == '__main__':
  main()
