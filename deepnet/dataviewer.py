"""Utility for computing data stats and adding to the data protocol buffers."""
import sys
from datahandler import *
from google.protobuf import text_format
# python dataviewer.py /ais/gobi3/u/nitish/mnist/mnist.pbtxt /ais/gobi3/u/nitish/mnist/mnist.pb

class DataViewer(object):
  def __init__(self, proto):
    self.proto = deepnet_pb2.Dataset()
    self.datasets = {}
    with open(proto, 'r') as f:
      text_format.Merge(f.read(), self.proto)

  def Load(self, name):
    this_set = next(d for d in self.proto.data if d.name == name)
    filenames = sorted(glob.glob(this_set.file_pattern))
    numdims = reduce(lambda a, x: a * x, this_set.dimensions)
    datasetsize = this_set.size
    disk_mem = datasetsize * 4 * numdims
    self.datasets[name] = Disk('disk', filenames, disk_mem, disk_mem/4, numdims=numdims)

  def AddStatsToProto(self, train_set_name, dest_sets):
    train_set = next(d for d in self.proto.data if d.name == train_set_name)
    d = self.datasets[train_set_name]
    d.ComputeDataStats()
    for dest_set in dest_sets:
      this_set = next(d for d in self.proto.data if d.name == dest_set)
      this_set.mean = d.mean.tostring()
      this_set.stddev = d.stddev.tostring()

  def SerializeProto(self, output_file):
    with open(output_file, 'wb') as f:
      f.write(self.proto.SerializeToString())

if __name__ == '__main__':
  data = DataViewer(sys.argv[1])
  source = 'train_data'
  dest = ['train_data', 'test_data' , 'valid_data']
  data.Load(source)
  data.AddStatsToProto(source, dest)
  data.SerializeProto(sys.argv[2])
