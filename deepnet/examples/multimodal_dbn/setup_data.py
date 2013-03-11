"""Sets up paths, separates out data with missing text."""
import numpy as np
from deepnet import datahandler as dh
from deepnet import util
import os, sys
from google.protobuf import text_format
import pdb
import glob

def EditPaths(data_pb, data_dir, gpu_mem, main_mem):
  data_pb.gpu_memory = gpu_mem
  data_pb.main_memory = main_mem
  data_pb.prefix = data_dir

def CreateMissingTextData(data_pb, data_pbtxt_file_z, data_pbtxt_file_nnz):
  """Some cases have text and some don't. This method separates them out."""
  prefix = data_pb.prefix
  data_pb_z = util.CopyDataset(data_pb)
  data_pb_nnz = util.CopyDataset(data_pb)
  data_pb_z.name = 'flickr_zero_text'
  data_pb_nnz.name = 'flickr_non_zero_text'
  del data_pb_z.data[:]
  del data_pb_nnz.data[:]

  for tag in ['labelled', 'unlabelled']:
    # Find the proto that describes text data.
    text_data = next(d for d in data_pb.data if d.name == 'text_%s' % tag)

    # Load the text data into a sparse matrix.
    text_data_file = os.path.join(prefix, text_data.file_pattern)
    data = dh.Disk.LoadSparse(text_data_file)

    # Find cases which have non-zero words.
    numwords = np.array(data.sum(axis=1)).reshape(-1)
    nnz_indices = np.where(numwords != 0)[0]
    z_indices = np.where(numwords == 0)[0]
    indices_file = os.path.join(prefix, 'text', 'indices_%s.npz' % tag)
    np.savez(indices_file, nnz_indices=nnz_indices, z_indices=z_indices)

    text_nnz_file = os.path.join('text', 'text_nnz_2000_%s.npz' % tag)
    dh.Disk.SaveSparse(os.path.join(prefix, text_nnz_file), data[nnz_indices])
    nnz = len(nnz_indices)

    # Separate images.
    image_data = next(d for d in data_pb.data if d.name == 'image_%s' % tag)
    numdims = np.prod(image_data.dimensions)
    image_z_dir = os.path.join('image', '%s_z' % tag)
    image_nnz_dir = os.path.join('image', '%s_nnz' % tag)

    data_writer_nnz = dh.DataWriter(['combined'],
                                    os.path.join(prefix, image_nnz_dir), '1G',
                                    [numdims], datasize=nnz)
    data_writer_z = dh.DataWriter(['combined'],
                                  os.path.join(prefix, image_z_dir), '1G',
                                  [numdims], datasize=image_data.size - nnz)
    end = 0
    img_files = glob.glob(os.path.join(data_pb.prefix, image_data.file_pattern))
    for img_file in sorted(img_files):
      print img_file
      img = np.load(img_file)
      start = end
      end = start + img.shape[0]
      nw = numwords[start:end]
      zero_text_images = img[np.where(nw == 0)[0]]
      non_zero_text_images = img[np.where(nw != 0)[0]]
      data_writer_z.Submit([zero_text_images])
      data_writer_nnz.Submit([non_zero_text_images])
    num_outputs_z = data_writer_z.Commit()
    num_outputs_nnz = data_writer_nnz.Commit()
    assert num_outputs_z[0] == image_data.size - nnz
    assert num_outputs_nnz[0] == nnz

    # Make data pbtxt for the new data.
    image_z = util.CopyData(image_data)
    image_nnz = util.CopyData(image_data)
    text_nnz = util.CopyData(text_data)

    image_z.file_pattern = os.path.join(image_z_dir, 'combined-*-of-*.npy')
    image_nnz.file_pattern = os.path.join(image_nnz_dir, 'combined-*-of-*.npy')
    text_nnz.file_pattern = text_nnz_file

    image_z.size = image_data.size - nnz
    image_nnz.size = nnz
    text_nnz.size = nnz

    data_pb_z.data.extend([image_z])
    data_pb_nnz.data.extend([image_nnz, text_nnz])

  with open(data_pbtxt_file_z, 'w') as f:
    text_format.PrintMessage(data_pb_z, f)

  with open(data_pbtxt_file_nnz, 'w') as f:
    text_format.PrintMessage(data_pb_nnz, f)

def EditTrainers(data_dir, model_dir, rep_dir, numsplits):
  tnames = ['train_CD_image_layer1.pbtxt',
            'train_CD_image_layer2.pbtxt',
            'train_CD_text_layer1.pbtxt',
            'train_CD_text_layer2.pbtxt',
            'train_CD_joint_layer.pbtxt']
  for tname in tnames:
    t_op_file = os.path.join('trainers', 'dbn', tname)
    t_op = util.ReadOperation(t_op_file)
    if 'layer1' in tname:
      t_op.data_proto_prefix = data_dir
    else:
      t_op.data_proto_prefix = rep_dir
    t_op.checkpoint_directory = model_dir
    with open(t_op_file, 'w') as f:
      text_format.PrintMessage(t_op, f)
  
  t_op_file = os.path.join('trainers', 'classifiers', 'baseclassifier.pbtxt')
  t_op = util.ReadOperation(t_op_file)
  for i in range(1, numsplits+1):
    t_op_file = os.path.join('trainers', 'classifiers', 'split_%d.pbtxt' % i)
    t_op.data_proto_prefix = rep_dir
    t_op.data_proto = os.path.join('split_%d' % i, 'data.pbtxt')
    t_op.checkpoint_prefix = model_dir
    t_op.checkpoint_directory = os.path.join('classifiers','split_%d' % i)  
    with open(t_op_file, 'w') as f:
      text_format.PrintMessage(t_op, f)

  # Change prefix in multimodal dbn model
  mnames = ['multimodal_dbn.pbtxt']
  for mname in mnames:
    model_file = os.path.join('models', mname)
    model = util.ReadModel(model_file)
    model.prefix = model_dir
    with open(model_file, 'w') as f:
      text_format.PrintMessage(model, f)


def main():
  data_dir = sys.argv[1]
  model_dir = sys.argv[2]
  rep_dir = sys.argv[3]
  gpu_mem = sys.argv[4]
  main_mem = sys.argv[5]
  numsplits = int(sys.argv[6])

  data_pbtxt_file = os.path.join(data_dir, 'flickr.pbtxt')
  data_pb = util.ReadData(data_pbtxt_file)
  EditPaths(data_pb, data_dir, gpu_mem, main_mem)
  with open(data_pbtxt_file, 'w') as f:
    text_format.PrintMessage(data_pb, f)
  EditTrainers(data_dir, model_dir, rep_dir, numsplits)

  data_pbtxt_file_z = os.path.join(data_dir, 'flickr_z.pbtxt')
  data_pbtxt_file_nnz = os.path.join(data_dir, 'flickr_nnz.pbtxt')
  if not os.path.exists(data_pbtxt_file_z):
    CreateMissingTextData(data_pb, data_pbtxt_file_z, data_pbtxt_file_nnz)
  data_pb = util.ReadData(data_pbtxt_file_z)
  EditPaths(data_pb, data_dir, gpu_mem, main_mem)
  with open(data_pbtxt_file_z, 'w') as f:
    text_format.PrintMessage(data_pb, f)
  data_pb = util.ReadData(data_pbtxt_file_nnz)
  EditPaths(data_pb, data_dir, gpu_mem, main_mem)
  with open(data_pbtxt_file_nnz, 'w') as f:
    text_format.PrintMessage(data_pb, f)


if __name__ == '__main__':
  main()
