"""Write a model protocol buffer to mat file."""
from deepnet import util
import numpy as np
import sys
import scipy.io

def Convert(model_file, output_file):
  model = util.ReadModel(model_file)
  params = {}
  for l in model.layer:
    for p in l.param:
      params['%s_%s' % (l.name, p.name)] = util.ParameterAsNumpy(p)
  for e in model.edge:
    for p in e.param:
      params['%s_%s_%s' % (e.node1, e.node2, p.name)] = util.ParameterAsNumpy(p)

  scipy.io.savemat(output_file, params, oned_as='column')

if __name__ == '__main__':
  Convert(sys.argv[1], sys.argv[2])

