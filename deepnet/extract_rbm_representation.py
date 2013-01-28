from neuralnet import *
from trainer import *

def ExtractRepresentations(model_file, train_op_file, layernames,
                           base_output_dir, memory='1G',
                           datasets=['train', 'validation', 'test'],
                           gpu_mem='2G', main_mem='30G'):

  if isinstance(model_file, str):
    model = util.ReadModel(model_file)
  else:
    model = model_file
  if isinstance(train_op_file, str):
    op = ReadOperation(train_op_file)
  else:
    op = train_op_file
  op.randomize = False
  op.verbose = True
  op.get_last_piece = True
  if not os.path.isdir(base_output_dir):
    os.makedirs(base_output_dir)

  net = CreateDeepnet(model, op, op)
  net.LoadModelOnGPU()
  net.SetUpData()

  data_pb = deepnet_pb2.Dataset()
  data_pb.name = model.name
  data_pb.gpu_memory = gpu_mem
  data_pb.main_memory = main_mem
  output_proto_file = os.path.join(base_output_dir, 'data.pbtxt')
  for dataset in datasets:
    output_dir = os.path.join(base_output_dir, dataset)
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    print 'Writing to %s' % output_dir
    size = net.WriteRepresentationToDisk(
      layernames, output_dir, memory=memory, dataset=dataset)
    # Write protocol buffer.
    tag = dataset
    """
    if dataset == 'train':
      tag = 'unlabelled'
    else:
      tag = 'labelled'
    """
    for i, lname in enumerate(layernames):
      layer = net.GetLayerByName(lname)
      data = data_pb.data.add()
      data.size = size[i]
      data.name = '%s_%s' % (lname, tag)
      data.file_pattern = os.path.join(output_dir, '%s-*-of-*.npy' % lname)
      data.dimensions.append(layer.state.shape[0])
  with open(output_proto_file, 'w') as f:
    text_format.PrintMessage(data_pb, f)

def main():
  board = LockGPU()
  model_file = sys.argv[1]
  train_op_file = sys.argv[2]
  layername = sys.argv[3]
  output_dir = sys.argv[4]
  datasets = ['train', 'validation', 'test']
  #datasets = ['test']
  gpu_mem = '2G'
  main_mem = '30G'
  if len(sys.argv) > 5:
    gpu_mem = sys.argv[5]
  if len(sys.argv) > 6:
    main_mem = sys.argv[6]

  ExtractRepresentations(model_file, train_op_file, [layername], output_dir,
                         datasets=datasets)
  FreeGPU(board)


if __name__ == '__main__':
  main()
