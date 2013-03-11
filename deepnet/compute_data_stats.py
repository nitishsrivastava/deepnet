"""Utility for computing data stats."""
# python compute_data_stats.py flickr.pbtxt flickr_stats.npz combined_unlabelled
import sys
from datahandler import *
from google.protobuf import text_format

class DataViewer(object):
  def __init__(self, proto_file):
    assert os.path.exists(proto_file)
    self.data_proto = util.ReadData(proto_file)

  def Load(self, name, batchsize=1000, typesize=4):
    data_proto = self.data_proto
    try:
      this_set = next(d for d in data_proto.data if d.name == name)
    except StopIteration as e:
      print 'No data called %s found in proto file.' % name
      raise e

    filenames = sorted(glob.glob(os.path.join(data_proto.prefix,
                                              this_set.file_pattern)))
    numdims = np.prod(np.array(this_set.dimensions))
    key = this_set.key
    self.numdims = numdims
    self.batchsize = batchsize
    datasetsize = this_set.size
    total_disk_space = datasetsize * numdims * typesize
    self.numbatches = datasetsize / batchsize
    max_cpu_capacity = min(total_disk_space, GetBytes(data_proto.main_memory))
    self.num_cpu_batches = max_cpu_capacity / (typesize * numdims * batchsize)
    cpu_capacity = self.num_cpu_batches * batchsize * numdims * typesize

    self.disk = Disk([filenames], [numdims], datasetsize, keys=[key])
    self.cpu_cache = Cache(self.disk, cpu_capacity, [numdims],
                           typesize = typesize, randomize=False)

  def Get(self):
    return self.cpu_cache.Get(self.batchsize)[0]

  def ComputeStats(self):
    numdims = self.numdims
    numbatches = self.numbatches
    means = np.zeros((numbatches, numdims))
    variances = np.zeros((numbatches, numdims))
    for i in range(numbatches):
      sys.stdout.write('\r%d of %d' % ((i + 1), numbatches))
      sys.stdout.flush()
      batch = self.Get()
      means[i] = batch.mean(axis=0)
      variances[i] = batch.var(axis=0)
    sys.stdout.write('\n')
    mean = means.mean(axis=0)
    std = np.sqrt(variances.mean(axis=0) + means.var(axis=0))
    mean_std = std.mean()
    std += (std == 0.0) * mean_std
    return {'mean': mean, 'std': std}

def Usage():
  print 'python %s <proto_file> <output_file> <data_name> ' % sys.argv[0] 
if __name__ == '__main__':
  if len(sys.argv) < 4:
    Usage()
    sys.exit()
  data_proto_file = sys.argv[1]
  outputfilename = sys.argv[2]
  data_name = sys.argv[3]
  dv = DataViewer(data_proto_file)
  dv.Load(data_name, batchsize=100)
  stats = dv.ComputeStats()
  np.savez(outputfilename, **stats)
