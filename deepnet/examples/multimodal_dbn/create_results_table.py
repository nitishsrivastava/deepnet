"""Collects results from multiple runs and puts them into a nice table."""
import sys
import numpy as np
from deepnet import util
import os

def main():
  path = sys.argv[1]
  numsplits = int(sys.argv[2])
  output_file = sys.argv[3]

  layers = ['image_input', 'image_hidden1', 'image_hidden2', 'joint_hidden',
            'text_hidden2', 'text_hidden1', 'text_input']
  maps = {}
  precs = {}
  for i in range(1, numsplits+1):
    for layer in layers:
      mfile = os.path.join(path, 'split_%d' % i, '%s_classifier_BEST' % layer)
      model = util.ReadModel(mfile)
      MAP = model.test_stat_es.MAP
      prec50 = model.test_stat_es.prec50
      if layer not in maps:
        maps[layer] = []
      if layer not in precs:
        precs[layer] = []
      maps[layer].append(MAP)
      precs[layer].append(prec50)

  f = open(output_file, 'w')
  f.write('\\begin{tabular}{|l|c|c|} \\hline \n')
  f.write('Layer & MAP & Prec@50 \\\\ \\hline\n')
  for layer in layers:
    lmap = np.array(maps[layer])
    lprec = np.array(precs[layer])
    f.write('%s & %.3f $\\pm$ %.3f & %.3f $\\pm$ %.3f \\\\ \n' % (layer,
            lmap.mean(), lmap.std(), lprec.mean(), lprec.std()))
  f.write('\\hline\n')
  f.write('\\end{tabular}\n')
  f.close()

if __name__ == '__main__':
  main()
