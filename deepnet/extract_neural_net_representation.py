"""Push the data through a network and get representations at each layer."""
from neuralnet import *
from trainer import *
import sys

def ExtractRepresentations(model_file, train_op_file, layernames,
                           base_output_dir, memory = '100M', skip_outputs=True,
                           datasets=['test'], gpu_mem='2G', main_mem='30G'):
  if isinstance(model_file, str):
    model = util.ReadModel(model_file)
  else:
    model = model_file
  if isinstance(train_op_file, str):
    op = ReadOperation(train_op_file)
  else:
    op = train_op_file
  if not os.path.isdir(base_output_dir):
    os.makedirs(base_output_dir)
  op.randomize = False
  op.get_last_piece = True
  net = CreateDeepnet(model, op, op)
  net.LoadModelOnGPU()
  net.SetUpData(skip_outputs=skip_outputs)

  data_pb = deepnet_pb2.Dataset()
  data_pb.name = model.name
  data_pb.gpu_memory = gpu_mem
  data_pb.main_memory =  main_mem
  output_proto_file = os.path.join(base_output_dir, 'data.pbtxt')
  for dataset in datasets:
    output_dir = os.path.join(base_output_dir, dataset)
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    print 'Writing to %s' % output_dir
    size = net.WriteRepresentationToDisk(
      layernames, output_dir, memory=memory, dataset=dataset)
    # Write protocol buffer.
    for i, lname in enumerate(layernames):
      if not size or size[i] == 0:
        continue
      layer = net.GetLayerByName(lname)
      data = data_pb.data.add()
      data.name = '%s_%s' % (lname, dataset)
      data.file_pattern = os.path.join(output_dir, '*-of-*.npy')
      data.size = size[i]
      data.dimensions.append(layer.state.shape[0])
  with open(output_proto_file, 'w') as f:
    text_format.PrintMessage(data_pb, f)

def Usage():
  print 'python %s <model_file> <train_op_file> <output_dir> <layer name1> [layer name2 [..]]' % sys.argv[0]

def main():
  if len(sys.argv) < 5:
    Usage()
    sys.exit(0)
  board = LockGPU()
  model_file = sys.argv[1]
  model = util.ReadModel(model_file)
  train_op_file = sys.argv[2]
  output_dir = sys.argv[3]
  layernames = sys.argv[4:]
  ExtractRepresentations(model_file, train_op_file, layernames, output_dir,
                         #memory='1G', datasets=['train', 'validation', 'test'])
                         memory='1G', datasets=['validation', 'test'])
  FreeGPU(board)


if __name__ == '__main__':
  main()
