from neuralnet import *
from trainer import *
#from extract_rbm_representation import *
def ExtractRepresentations(model_file, train_op_file, layernames,
                           base_output_dir, memory = '100M'):
  model = util.ReadModel(model_file)
  op = ReadOperation(train_op_file)
  op.randomize = False
  net = CreateDeepnet(model, op, op)
  net.LoadModelOnGPU()
  net.SetUpData()

  data_pb = deepnet_pb2.Dataset()
  data_pb.name = model.name
  data_pb.gpu_memory = '5G'
  data_pb.main_memory =  '30G'
  output_proto_file = os.path.join(base_output_dir, 'data.pbtxt')
  for dataset in ['train', 'validation', 'test']:
    output_dir = os.path.join(base_output_dir, dataset)
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
  layernames = ['joint_hidden', 'text_hidden2', 'text_hidden1', 'image_hidden2',
                'image_hidden1', 'image_input_layer', 'text_input_layer']
  if len(sys.argv) > 3:
    output_d = sys.argv[3]
  else:
    output_d = 'dbn_reps'
  output_dir = os.path.join(prefix, output_d, '%s_LAST' % model.name)
  #model_file = os.path.join(prefix, 'models', '%s_LAST' % model.name)
  model_file = sys.argv[1]
  ExtractRepresentations(model_file, train_op_file, layernames, output_dir, memory='1G')
  FreeGPU()


if __name__ == '__main__':
  main()
