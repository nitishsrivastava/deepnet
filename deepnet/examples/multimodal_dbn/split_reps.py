import numpy as np
import glob, os, sys
from deepnet import deepnet_pb2
from deepnet import util
from google.protobuf import text_format 

def DumpDataSplit(data, output_dir, name, dataset_pb, stats_file):
  data_pb = dataset_pb.data.add()
  output_file_name = os.path.join(output_dir, name)
  np.save(output_file_name, data)
  data_pb.name = name
  data_pb.file_pattern = '%s.npy' % output_file_name
  data_pb.size = data.shape[0]
  if stats_file:
    data_pb.stats_file = stats_file
  data_pb.dimensions.append(data.shape[1])

def DumpLabelSplit(data, output_dir, name, dataset_pb):
  data_pb = dataset_pb.data.add()
  output_file_name = os.path.join(output_dir, name)
  np.save(output_file_name, data)
  data_pb.name = name
  data_pb.file_pattern = '%s.npy' % output_file_name
  data_pb.size = data.shape[0]
  data_pb.dimensions.append(data.shape[1])

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

def MakeDict(data_pbtxt):
  data_pb = util.ReadData(data_pbtxt)
  rep_dict = {}
  stats_files = {}
  for data in data_pb.data:
    rep_dict[data.name] = Load(data.file_pattern)
    stats_files[data.name] = data.stats_file
  return rep_dict, stats_files

def main():
  data_pbtxt = sys.argv[1]
  output_dir = sys.argv[2]
  prefix = sys.argv[3]
  r = int(sys.argv[4])
  gpu_mem = sys.argv[5]
  main_mem = sys.argv[6]
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

  rep_dict, stats_files = MakeDict(data_pbtxt)
  reps = rep_dict.keys()

  indices_file = os.path.join(prefix, 'splits', 'train_indices_%d.npy' % r)
  if os.path.exists(indices_file):
    train = np.load(indices_file)
    valid = np.load(os.path.join(prefix, 'splits', 'valid_indices_%d.npy' % r))
    test = np.load(os.path.join(prefix, 'splits', 'test_indices_%d.npy' % r))
  else:
    print 'Creating new split.'
    indices = np.arange(25000)
    np.random.shuffle(indices)
    train = indices[:10000]
    valid = indices[10000:15000]
    test = indices[15000:]
    np.save(os.path.join(prefix, 'splits', 'train_indices_%d.npy' % r), train)
    np.save(os.path.join(prefix, 'splits', 'valid_indices_%d.npy' % r), valid)
    np.save(os.path.join(prefix, 'splits', 'test_indices_%d.npy' % r), test)

    
  print 'Splitting data'
  dataset_pb = deepnet_pb2.Dataset()
  dataset_pb.name = 'flickr_split_%d' % r
  dataset_pb.gpu_memory = gpu_mem
  dataset_pb.main_memory = main_mem
  for rep in reps:
    data = rep_dict[rep]
    stats_file = stats_files[rep]
    DumpDataSplit(data[train], output_dir, 'train_%s' % rep, dataset_pb, stats_file)
    DumpDataSplit(data[valid], output_dir, 'valid_%s' % rep, dataset_pb, stats_file)
    DumpDataSplit(data[test], output_dir, 'test_%s' % rep, dataset_pb, stats_file)

  print 'Splitting labels'
  labels = np.load(os.path.join(prefix, 'labels.npy')).astype('float32')
  DumpLabelSplit(labels[train,], output_dir, 'train_labels', dataset_pb)
  DumpLabelSplit(labels[valid,], output_dir, 'valid_labels', dataset_pb)
  DumpLabelSplit(labels[test,], output_dir, 'test_labels', dataset_pb)

  #d = 'indices'
  #np.save(os.path.join(output_dir, 'train_%s.npy' % d), train)
  #np.save(os.path.join(output_dir, 'valid_%s.npy' % d), valid)
  #np.save(os.path.join(output_dir, 'test_%s.npy' % d), test)

  with open(os.path.join(output_dir, 'data.pbtxt'), 'w') as f:
    text_format.PrintMessage(dataset_pb, f)

  print 'Output written in directory %s' % output_dir

if __name__ == '__main__':
  main()
