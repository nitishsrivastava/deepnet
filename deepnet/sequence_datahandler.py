import datahandler as dh
import pdb
import sys

class SequenceDisk(dh.Disk):
  def __init__(self, *args, **kwargs):
    super(SequenceDisk, self).__init__(*args, **kwargs)
    self.disable_chunk_split = True
    self.collect_chunk_boundaries = True
    self.left_window = kwargs.get('left_window', [])
    self.right_window = kwargs.get('right_window', [])

  def LoadSequence(self, inputfile):
    """Loads a squence file stored in a pickle."""

    data_pkl = dh.Disk.LoadPickle(inputfile)
    data_list = []
    for i, key in enumerate(self.keys):
      seq = data_pkl[key].T
      if len(seq.shape) == 1:
        seq = seq.reshape(-1, 1)

      # Add padding.
      lw = self.left_window[i]
      rw = self.right_window[i]
      if lw > 0 or rw > 0:
        padded_length = lw + seq.shape[0] + rw
        numdims = seq.shape[1]
        seq_padded = dh.np.zeros((padded_length, numdims))

        if lw > 0:
          seq_padded[0:lw,:] = dh.np.tile(seq[0], (lw, 1))

        start = lw
        end = seq.shape[0] + lw
        seq_padded[start:end,:] = seq

        if rw > 0:
          seq_padded[end:end+rw,:] = dh.np.tile(seq[-1], (rw, 1))
        data_list.append(seq_padded)
      else:
        data_list.append(seq)

    return data_list

  def Get(self, batchsize):
    """Reads data from disk.
    Args:
      batchsize: Number of data points to read.
    Returns:
      A list of numpy arrays each with batchsize rows. Each element of the list
      is one data modality.
    """
    assert self.num_data <= 2

    i = 0
    numdims = self.numdim_list[i]
    filename_list = self.filenames[i]
    num_files = self._num_file_list[i]
    current_file = (self.last_read_file[i] + 1) % num_files

    data_list = []
    boundaries = []
    datasize = [0]*self.num_data  # Number of rows of data filled up.
    for i in range(self.num_data):
      boundaries.append([])
      data_list.append(dh.np.zeros((batchsize, self.numdim_list[i]), dtype='float32'))

    # Read data from disk.
    while(datasize[0] < batchsize):
      if self.last_read_file[0] != current_file:
        if self.verbose:
          sys.stdout.write('\rLoading %s ...' % filename_list[current_file])
          sys.stdout.flush()
        this_chunk = self.LoadSequence(filename_list[current_file])
        self.last_read_chunk[0] = this_chunk
        self.last_read_file[0] = current_file
      else:
        this_chunk = self.last_read_chunk[0]
      is_full = False
      for i, d in enumerate(this_chunk):
        chunk_size = d.shape[0]
        if chunk_size + datasize[i] > batchsize:
          is_full = True
      if is_full:
        for i in range(len(this_chunk)):
          data_list[i] = data_list[i][:datasize[i]]
        break
      else:
        for i, d in enumerate(this_chunk):
          lw = self.left_window[i]
          rw = self.right_window[i]
          cs = d.shape[0]
          ds = datasize[i]
          data_list[i][ds : ds + cs] = d
          # if lw + rw > 0:
          #   valid_boundaries = range(ds + lw, ds + cs - rw)
          boundaries[i].append(cs)
          datasize[i] += cs
      current_file = (current_file + 1) % num_files
    if self.verbose:
      sys.stdout.write('\n')
    return data_list, boundaries

class SequenceCache(dh.Cache):
  def __init__(self, *args, **kwargs):
    super(SequenceCache, self).__init__(*args, **kwargs)
    self.left_window = kwargs.get('left_window', [])
    self.right_window = kwargs.get('right_window', [])
    self.data_len = len(self.left_window)
    self._pos = [0] * self.data_len
    self._relpos = [0] * self.data_len
    self._utt = [0] * self.data_len
    max_padding = 0
    max_padding_i = 0
    for i in range(self.data_len):
      lw = self.left_window[i]
      rw = self.right_window[i]
      if lw + rw > max_padding:
        max_padding_i = i
        max_padding = lw + rw
    self.max_padding_i = max_padding_i

  def LoadData(self):
    if self.data == [] or self._maxpos < self.parent._maxpos:
      data, boundaries = self.parent.Get(self._maxpos)
      self.data = data
      self.boundaries = boundaries
      self.datasize = self.data[self.max_padding_i].shape[0]

  def Get(self, batchsize, mult_of):
    max_i = self.max_padding_i
    if self._pos[max_i] == self.datasize:
      for i in range(self.data_len):
        self._pos[i] = 0
    if self._pos[max_i] == 0:
      self.LoadData()
    
    max_lw = self.left_window[max_i]
    max_rw = self.right_window[max_i]

    startpos = self._pos[max_i]  # pos from start of in-memory data.
    start_relpos = self._relpos[max_i]  # Relative pos from start of utterance.
    start_utt = self._utt[max_i]  # Current utterance.
    bd = self.boundaries[max_i]

    endpos = min(startpos + batchsize, self.datasize)

    # Find number of valid indices between start_pos and end_pos.
    utt = start_utt
    relpos = start_relpos
    num_valid = 0
    pos = 0
    while pos < endpos - startpos:
      f = bd[utt]
      if pos + f - relpos > endpos - startpos:
        remaining = endpos - startpos - pos
        if relpos + remaining > f - max_rw:
          remaining = f - rw - relpos
        if relpos > max_lw:
          num_valid_in_this_utt = remaining
        else:
          num_valid_in_this_utt = max(0, relpos + remaining - lw)
        num_valid += num_valid_in_this_utt
        pos = endpos - startpos
        relpos += remaining
      else:
        if relpos < max_lw:
          num_valid_in_this_utt = f - max_lw - max_rw
        elif relpos > f - max_rw:
          num_valid_in_this_utt = 0
        else:
          num_valid_in_this_utt = f - relpos - max_rw
        num_valid += num_valid_in_this_utt
        pos += f - relpos
        relpos = 0
        utt += 1
    num_valid = (num_valid / mult_of) * mult_of

    batch = []
    indices = []
    for i in range(self.data_len):
      startpos = self._pos[i]  # pos from start of in-memory data.
      relpos = self._relpos[i]  # Relative pos from start of utterance.
      utt = self._utt[i]  # Current utterance.
      lw = self.left_window[i]
      rw = self.right_window[i]

      this_valid = 0
      pos = 0
      bd = self.boundaries[i]
      this_indices = []
      while(this_valid < num_valid):
        f = bd[utt]
        if relpos < lw:
          num_valid_in_this_utt = f - lw - rw
          start_valid = pos + lw
        elif relpos > f - rw:
          num_valid_in_this_utt = 0
          start_valid = pos
        else:
          num_valid_in_this_utt = f - relpos - rw
          start_valid = pos
        if this_valid + num_valid_in_this_utt > num_valid:
          num_valid_needed = num_valid - this_valid
          this_indices.extend(range(
            start_valid, start_valid + num_valid_needed))
          pos += lw + num_valid_needed + rw
          relpos = lw + num_valid_needed
          break
        else:
          this_valid += num_valid_in_this_utt
          this_indices.extend(range(
            start_valid, start_valid + num_valid_in_this_utt))
          pos += f - relpos
          relpos = 0
          utt += 1
      batch.append(self.data[i][startpos:startpos + pos])
      indices.append(this_indices)
      self._utt[i] = utt
      self._pos[i] += pos 
      self._relpos[i] = relpos
    return batch, indices


class SequenceGPUCache(dh.GPUCache):
  """Manager for a cache that stores sequential data."""

  def __init__(self, *args, **kwargs):
    super(SequenceGPUCache, self).__init__(*args, **kwargs)
    self.left_window = kwargs.get('left_window', [])
    self.right_window = kwargs.get('right_window', [])
    self.batchsize = kwargs.get('batchsize')
    batchsize = self.batchsize
    #self.indices = dh.cm.CUDAMatrix(dh.np.arange(batchsize).reshape(1, -1))
    self.batches = []
    self.templates = []
    self.window_sizes = []
    self.batch_indices = []
    self.data_len = len(self.left_window)
    self.AllocateBatchsizeDepedentMemory(batchsize)

    self.data = []
    self.valid_indices = []
    self.empty = True
    for i in range(self.data_len):
      self.data.append(dh.cm.CUDAMatrix(dh.np.zeros((self.numdim_list[i], self._maxpos))))
      self.valid_indices.append(dh.cm.CUDAMatrix(dh.np.zeros((1, self._maxpos))))

  def AllocateBatchsizeDepedentMemory(self, batchsize):
    self.batches = []
    self.templates = []
    self.window_sizes = []
    self.batch_indices = []
    for i in range(self.data_len):
      l = self.left_window[i]
      r = self.right_window[i]
      window_size = 1 + l + r
      numdims = self.numdim_list[i]
      batch = dh.cm.empty((numdims * window_size, batchsize))
      window = dh.np.arange(-l, r + 1).reshape(-1, 1)
      template = dh.cm.CUDAMatrix(dh.np.tile(window, (1, batchsize)))
      self.batches.append(batch)
      self.templates.append(template)
      self.window_sizes.append(window_size)
      self.batch_indices.append(dh.cm.empty(template.shape))


  def ShuffleData(self):
    indices = dh.np.arange(self.datasize)
    dh.np.random.shuffle(indices)
    indices1 = indices[:self.datasize/2]
    indices2 = indices[self.datasize/2:2*(self.datasize/2)]
    indices1_gpu = dh.cm.CUDAMatrix(indices1.reshape(1, -1))
    indices2_gpu = dh.cm.CUDAMatrix(indices2.reshape(1, -1))
    for d in self.valid_indices:
      d.swap_columns(indices1_gpu, indices2_gpu, target=d)
    indices1_gpu.free_device_memory()
    indices2_gpu.free_device_memory()

  def Get(self, batchsize, get_last_piece=False):
    """Return 'batchsize' data points from the cache."""
    skip = False
    if self._pos == self.datasize:
      self._pos = 0
    if self._pos == 0:
      if self.empty or self._maxpos < self.parent._maxpos:
        if get_last_piece:
          self.LoadData(1)
        else:
          self.LoadData(batchsize)
        self.empty = False
      if self.randomize:
        self.ShuffleData()
    start = self._pos
    end = self._pos + batchsize
    if end > self.datasize:
      end = self.datasize
      skip = not get_last_piece
    self._pos = end
    if skip:
      return self.Get(batchsize, get_last_piece=get_last_piece)
    else:
      for i, d in enumerate(self.data):
        centers = self.valid_indices[i].slice(start, end)
        self.ExtractWindows(d, centers, i)
      return self.batches

  def ExtractWindows(self, d, centers, i):
    """Extracts window around the indices in 'centers' from d."""
    batchsize = centers.shape[1] 
    if batchsize != self.batches[i].shape[1]:
      self.AllocateBatchsizeDepedentMemory(batchsize)
    batch = self.batches[i]
    template = self.templates[i]
    batch_indices = self.batch_indices[i]
    window_size = self.window_sizes[i]
    numdims = self.numdim_list[i]

    batch_indices.reshape((window_size, batchsize))
    template.add_row_vec(centers, target=batch_indices)
    batch_indices.reshape((1, window_size * batchsize))
    batch.reshape((numdims, window_size * batchsize))
    d.select_columns(batch_indices, target=batch)
    batch.reshape((numdims * window_size, batchsize))

  def LoadData(self, batchsize):
    data_cpu, indices_cpu = self.parent.Get(self._maxpos, batchsize)
    datasize = len(indices_cpu[0])
    self.datasize = datasize
    for i, d in enumerate(data_cpu):
      mat = d.T
      self.data[i].overwrite(mat)
      self.valid_indices[i].overwrite(dh.np.array(
        indices_cpu[i]).reshape(1, -1))
    self.Normalize()

  def Normalize(self):
    for i, batch in enumerate(self.data):
      if self.normalize[i]:
        mean = self.means[i]
        std = self.stds[i]
        window_size = self.window_sizes[i]
        batchsize = self.batchsize
        numdims = self.numdim_list[i]
        batch.add_col_mult(mean, mult=-1.0)
        batch.div_by_col(std)
