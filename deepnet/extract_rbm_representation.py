from neuralnet import *
from trainer import *

def ExtractRepresentations(model_file, train_op_file, layernames,
                           base_output_dir, memory='10G',
                           datasets=['validation', 'test', 'train'],
                           gpu_mem='2G', main_mem='30G', data_proto=None):
  if isinstance(model_file, str):
    model = util.ReadModel(model_file)
  else:
    model = model_file
  if isinstance(train_op_file, str):
    op = ReadOperation(train_op_file)
  else:
    op = train_op_file
  if data_proto:
    op.data_proto = data_proto
  op.randomize = False
  op.verbose = False
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
  data_pb.prefix = base_output_dir
  output_proto_file = os.path.join(base_output_dir, 'data.pbtxt')
  for dataset in datasets:
    output_dir = os.path.join(base_output_dir, dataset)
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    print 'Writing to %s' % output_dir
    size = net.WriteRepresentationToDisk(
      layernames, output_dir, memory=memory, dataset=dataset, input_recon=True)
    # Write protocol buffer.
    tag = dataset
    if size is None:
      continue
    for i, lname in enumerate(layernames):
      layer = net.GetLayerByName(lname)
      data = data_pb.data.add()
      data.size = size[i]
      data.name = '%s_%s' % (lname, tag)
      data.file_pattern = os.path.join(dataset, '%s-*-of-*.npy' % lname)
      data.dimensions.append(layer.state.shape[0])
  with open(output_proto_file, 'w') as f:
    text_format.PrintMessage(data_pb, f)

def main():
  board = LockGPU()
  model_file = sys.argv[1]
  train_op_file = sys.argv[2]
  layernames = sys.argv[3].split()
  output_dir = sys.argv[4]
  datasets = ['validation', 'test', 'train']
  #datasets = ['validation', 'test']
  #datasets = ['test']
  gpu_mem = '2G'
  main_mem = '30G'
  data_proto = None
  if len(sys.argv) > 5:
    gpu_mem = sys.argv[5]
  if len(sys.argv) > 6:
    main_mem = sys.argv[6]
  if len(sys.argv) > 7:
    data_proto = sys.argv[7]

  ExtractRepresentations(model_file, train_op_file, layernames, output_dir,
                         datasets=datasets, gpu_mem=gpu_mem, main_mem=main_mem,
                         data_proto=data_proto)
  FreeGPU(board)


if __name__ == '__main__':
  main()
