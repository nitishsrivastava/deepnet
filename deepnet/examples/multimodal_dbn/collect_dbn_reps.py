"""Collects Multimodal-DBN representations.
This script combines representations created for all inputs, whether missing
text or not in one place to be used for classification/retrieval.
"""
import numpy as np
import sys
import os
from deepnet import deepnet_pb2
from deepnet import util
import glob
from deepnet import datahandler as dh
import pdb
from google.protobuf import text_format

def main():
  model_file = sys.argv[1]
  base_output_dir = sys.argv[2]
  rep_dir = sys.argv[3]
  prefix = sys.argv[4]
  gpu_mem = sys.argv[5]
  main_mem = sys.argv[6]
  model = util.ReadModel(model_file)
  data_pb = deepnet_pb2.Dataset()
  data_pb.name = model.name
  data_pb.gpu_memory = gpu_mem
  data_pb.main_memory = main_mem
  output_dir = os.path.join(base_output_dir, 'validation')
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  output_proto_file = os.path.join(base_output_dir, 'data.pbtxt')

  # IMAGE PATHWAY
  img_input_pbtxt = os.path.join(prefix, 'flickr.pbtxt')
  img_hidden1_pbtxt = os.path.join(rep_dir, 'image_rbm1_LAST', 'data.pbtxt')
  img_hidden2_pbtxt = os.path.join(rep_dir, 'image_rbm2_LAST', 'data.pbtxt')
 
  # TEXT PATHWAY
  text_input_pbtxt = os.path.join(prefix, 'flickr_nnz.pbtxt')
  text_hidden1_pbtxt = os.path.join(rep_dir, 'text_rbm1_LAST', 'data.pbtxt')
  text_hidden2_pbtxt = os.path.join(rep_dir, 'text_rbm2_LAST', 'data.pbtxt')
  text_pbtxt_z = os.path.join(rep_dir, 'generated_text', 'data.pbtxt')
  
  joint_pbtxt = os.path.join(rep_dir, 'joint_rbm_LAST', 'data.pbtxt')

  
  img_input_pb = util.ReadData(img_input_pbtxt)
  data = next(d for d in img_input_pb.data if d.name == 'image_labelled')
  data.file_pattern = os.path.join(img_input_pb.prefix, data.file_pattern)
  data.stats_file = os.path.join(img_input_pb.prefix, data.stats_file)
  data.name = 'image_input'
  data_pb.data.extend([data])

  img_hidden1_pb = util.ReadData(img_hidden1_pbtxt)
  data = next(d for d in img_hidden1_pb.data if d.name == 'image_hidden1_validation')
  data.file_pattern = os.path.join(img_hidden1_pb.prefix, data.file_pattern)
  data.name = 'image_hidden1'
  data_pb.data.extend([data])

  img_hidden2_pb = util.ReadData(img_hidden2_pbtxt)
  data = next(d for d in img_hidden2_pb.data if d.name == 'image_hidden2_validation')
  data.file_pattern = os.path.join(img_hidden2_pb.prefix, data.file_pattern)
  data.name = 'image_hidden2'
  data_pb.data.extend([data])
  
  indices_file = os.path.join(prefix, 'text', 'indices_labelled.npz')
  indices = np.load(indices_file)
  nnz_indices = indices['nnz_indices']
  z_indices = indices['z_indices']

  text_pb_z = util.ReadData(text_pbtxt_z)
  text_input_pb = util.ReadData(text_input_pbtxt)
  data_nnz = next(d for d in text_input_pb.data if d.name == 'text_labelled')
  data_z = next(d for d in text_pb_z.data if d.name == 'text_input_layer_validation')
  output_file = os.path.join(output_dir, 'text_input-00001-of-00001.npy')
  data = Merge(data_nnz, data_z, nnz_indices, z_indices, text_pb_z.prefix, text_input_pb.prefix, 'text_input', output_file)
  data_pb.data.extend([data])

  text_hidden1_pb = util.ReadData(text_hidden1_pbtxt)
  data_nnz = next(d for d in text_hidden1_pb.data if d.name == 'text_hidden1_validation')
  data_z = next(d for d in text_pb_z.data if d.name == 'text_hidden1_validation')
  output_file = os.path.join(output_dir, 'text_hidden1-00001-of-00001.npy')
  data = Merge(data_nnz, data_z, nnz_indices, z_indices, text_pb_z.prefix, text_hidden1_pb.prefix, 'text_hidden1', output_file)
  data_pb.data.extend([data])

  text_hidden2_pb = util.ReadData(text_hidden2_pbtxt)
  data_nnz = next(d for d in text_hidden2_pb.data if d.name == 'text_hidden2_validation')
  data_z = next(d for d in text_pb_z.data if d.name == 'text_hidden2_validation')
  output_file = os.path.join(output_dir, 'text_hidden2-00001-of-00001.npy')
  data = Merge(data_nnz, data_z, nnz_indices, z_indices, text_pb_z.prefix, text_hidden2_pb.prefix, 'text_hidden2', output_file)
  data_pb.data.extend([data])

  joint_pb = util.ReadData(joint_pbtxt)
  data_nnz = next(d for d in joint_pb.data if d.name == 'joint_hidden_validation')
  data_z = next(d for d in text_pb_z.data if d.name == 'joint_hidden_validation')
  output_file = os.path.join(output_dir, 'joint_hidden-00001-of-00001.npy')
  data = Merge(data_nnz, data_z, nnz_indices, z_indices, text_pb_z.prefix, joint_pb.prefix, 'joint_hidden', output_file)
  data_pb.data.extend([data])

  with open(output_proto_file, 'w') as f:
    text_format.PrintMessage(data_pb, f)

def Load(file_pattern):
  data = None
  for f in sorted(glob.glob(file_pattern)):
    ext = os.path.splitext(f)[1]
    if ext == '.npy':
      this_data = np.load(f)
    elif ext == '.npz':
      this_data = dh.Disk.LoadSparse(f).toarray()
    else:
      raise Exception('unknown data format.')
    if data is None:
      data = this_data
    else:
      data = np.concatenate((data, this_data))
  return data

def Merge(data_nnz, data_z, indices_nnz, indices_z, prefix_z, prefix_nnz, name, output_file):
  data_nnz = Load(os.path.join(prefix_nnz, data_nnz.file_pattern))
  data_z = Load(os.path.join(prefix_z, data_z.file_pattern))
  assert data_nnz.shape[1] == data_z.shape[1], 'Dimension mismatch.'
  size = data_nnz.shape[0] + data_z.shape[0]
  numdims = data_nnz.shape[1]
  data = np.zeros((size, numdims), dtype=np.float32)
  data[indices_nnz] = data_nnz
  data[indices_z] = data_z
  np.save(output_file, data)

  data = deepnet_pb2.Dataset.Data()
  data.name = name
  data.size = size
  data.dimensions.extend([numdims])
  data.file_pattern = output_file

  return data

if __name__ == '__main__':
  main()

