import datahandler
import sys
import cudamat as cm
import numpy as np
sys.path.append('/u/ndjaitly/workspace/Common')
sys.path.append('/u/ndjaitly/workspace/cudamat_ext')
import spec_data

class NavdeepDataHandler(datahandler.DataHandler):

  def __init__(self, name, proto, op, hyp, typesize=4,
               boundary_proto=None, permutation_link=None):

    self.name = name
    batch_size = op.batchsize
    if permutation_link is None:
      db_name = proto.name.split('_')[0]  # Hack!
      db_path = proto.file_pattern
      if db_name == 'validation':
        if 'TIMIT' in db_path:
          db_name = 'dev'
        else:
          db_name = 'test_dev93'
      if db_name == 'train':
        if 'TIMIT' in db_path:
          pass
        else:
          db_name = 'train_si284'
      left = hyp.left_window
      right = hyp.right_window
      num_frames = 1 + left + right
      normalize = True
      self.data_src = spec_data.SpecData(db_name, db_path, 
                                         num_frames, batch_size,
                                         normalize_data=normalize,
                                         num_sentences_in_mem=400,
                                         return_label_indices=True)
      self.data_dict = self.data_src.get_data_ref()
      self.data = self.data_dict['data']
      self.is_controller = True
      self.data_iter = self.data_src.get_iter()
    else:
      self.data = permutation_link.data_dict['labels']
      self.is_controller = False

    self.num_batches = proto.size / batch_size;
    print '%s %d batches of shape %s' % (self.name, self.num_batches,
                                         self.data.shape.__str__())

  def Get(self):
    try:
      if self.is_controller:
        self.data_iter.next()
      return self.data
    except StopIteration:
      self.data_iter = self.data_src.get_iter()
      return self.Get()
