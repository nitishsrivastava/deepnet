from neuralnet import *
from trainer import *

def ExtractRepresentations(model_file, train_op_file, layernames,
                           base_output_dir, memory = '100M', datasets=['test']):

  if isinstance(model_file, str):
    model = util.ReadModel(model_file)
  else:
    model = model_file
  if isinstance(train_op_file, str):
    op = ReadOperation(train_op_file)
  else:
    op = train_op_file
  op.randomize = False
  if not os.path.isdir(base_output_dir):
    os.makedirs(base_output_dir)

  net = CreateDeepnet(model, op, op)
  net.LoadModelOnGPU()
  net.SetUpData()

  data_pb = deepnet_pb2.Dataset()
  data_pb.name = model.name
  data_pb.gpu_memory = '5G'
  data_pb.main_memory =  '30G'
  output_proto_file = os.path.join(base_output_dir, 'data.pbtxt')
  for dataset in datasets:
    output_dir = os.path.join(base_output_dir, dataset)
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    print 'Writing to %s' % output_dir
    size = net.WriteRepresentationToDisk(
      layernames, output_dir, memory=memory, dataset=dataset)
    if size is None:
      continue
    # Write protocol buffer.
    if dataset == 'train':
      tag = 'unlabelled'
    else:
      tag = 'labelled'
    for lname in layernames:
      layer = net.GetLayerByName(lname)
      data = data_pb.data.add()
      data.name = '%s_%s' % (lname, tag)
      data.file_pattern = os.path.join(output_dir, '*-of-*.npy')
      data.size = size
      data.dimensions.append(layer.state.shape[0])
  with open(output_proto_file, 'w') as f:
    text_format.PrintMessage(data_pb, f)

def main():
  LockGPU()
  prefix = '/ais/gobi3/u/nitish/flickr'
  model = util.ReadModel(sys.argv[1])
  train_op_file = sys.argv[2]
  layernames = [sys.argv[3]]
  if len(sys.argv) > 4:
    output_d = sys.argv[4]
  else:
    output_d = 'rbm_reps'
  output_dir = os.path.join(prefix, output_d, '%s_LAST' % model.name)
  model_file = os.path.join(prefix, 'models', '%s_LAST' % model.name)
  ExtractRepresentations(model_file, train_op_file, layernames, output_dir)
  FreeGPU()


if __name__ == '__main__':
  main()
