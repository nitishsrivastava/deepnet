"""Do inference in deepnet models."""
from neuralnet import *
from trainer import *

def DoInference(model_file, train_op_file, base_output_dir, layernames,
                layernames_to_unclamp, memory='1G', method='gibbs',
                steps=10, datasets=['validation', 'test'], gpu_mem='2G',
                main_mem='30G', data_proto=None):
  model = util.ReadModel(model_file)
  op = ReadOperation(train_op_file)
  op.randomize = False
  op.get_last_piece = True
  if data_proto:
    op.data_proto = data_proto
  net = CreateDeepnet(model, op, op)
  net.LoadModelOnGPU()
  net.SetUpData(skip_layernames=layernames_to_unclamp)

  data_pb = deepnet_pb2.Dataset()
  data_pb.name = model.name
  data_pb.gpu_memory = gpu_mem
  data_pb.main_memory =  main_mem
  output_proto_file = os.path.join(base_output_dir, 'data.pbtxt')
  for dataset in datasets:
    output_dir = os.path.join(base_output_dir, dataset)
    print 'Writing to %s' % output_dir
    size = net.Inference(steps, layernames, layernames_to_unclamp, output_dir,
                         memory=memory, dataset=dataset, method=method)
    if size is None:
      continue
    # Write protocol buffer.
    for lname in layernames:
      layer = net.GetLayerByName(lname)
      data = data_pb.data.add()
      data.name = '%s_%s' % (lname, dataset)
      data.file_pattern = os.path.join(output_dir, '%s-*-of-*.npy' % lname)
      data.size = size
      data.dimensions.append(layer.state.shape[0])
  with open(output_proto_file, 'w') as f:
    text_format.PrintMessage(data_pb, f)

def main():
  LockGPU()
  prefix = '/ais/gobi3/u/nitish/flickr'
  model = util.ReadModel(sys.argv[1])
  train_op_file = sys.argv[2]
  layernames = ['joint_hidden', 'text_hidden2', 'text_hidden1',
                'text_input_layer']
  layernames_to_unclamp = ['text_input_layer', 'text_hidden2']
  method = 'gibbs'
  steps = 10
  output_d = 'dbn_inference'

  output_dir = os.path.join(prefix, output_d, '%s_LAST' % model.name)
  model_file = sys.argv[1]
  DoInference(model_file, train_op_file, output_dir, layernames,
              layernames_to_unclamp, memory = '1G', method=method,
              steps=steps)
  FreeGPU()


if __name__ == '__main__':
  main()
