from deepnet import util
from deepnet import deepnet_pb2
import cPickle as pickle
from choose_matrix_library import *
import glob
import numpy as np
import os.path
import scipy.sparse as sp
import pdb
import gzip
import random

class Disk(object):
  """A Disk access manager."""
  def __init__(self, filenames, numdim_list, total_size, keys=[], verbose=False,
              **kwargs):
    """Initializes a Disk object.

    Args:
      filenames: List of list of filenames.
      numdim_list: List of integers that represent the dimensionality of the
        data.
      total_size: Number of data points in the dataset (sum over all files).
      verbose: If True, will print out details of what is happening.
    """
    assert len(filenames) == len(numdim_list)
    self.num_data = len(filenames)
    self.numdim_list = numdim_list
    self.filenames = filenames
    self._num_file_list = [len(filename_list) for filename_list in filenames]
    self._maxpos = total_size
    self.verbose = verbose
    self.left_overs = [None]*self.num_data
    self.last_read_chunk = [None]*self.num_data
    self.last_read_file = [-1]*self.num_data
    self.data = [None]*self.num_data
    if keys:
      self.keys = keys
    else:
      self.keys = [None]*self.num_data


  def AddSparseData(self, data, chunk):
    """Appends chunk to data."""
    if data is None:
      return chunk
    else:
      return sp.vstack((data, chunk)).tocsr()

  def Get(self, batchsize):
    """Reads data from disk.
    Args:
      batchsize: Number of data points to read.
    Returns:
      A list of numpy arrays each with batchsize rows. Each element of the list
      is one data modality.
    """
    data_list = []
    for i in range(self.num_data):
      key = self.keys[i]
      numdims = self.numdim_list[i]
      filename_list = self.filenames[i]
      num_files = self._num_file_list[i]
      current_file = (self.last_read_file[i] + 1) % num_files
      sparse = os.path.splitext(filename_list[current_file])[1] == '.npz'

      # Allocate memory for storing data that will come from the disk.
      if sparse:
        # Sparse matrices do not allow slice assignment, so we will stack them
        # as they come.
        data = None
      else:
        if self.data[i] is None:
          self.data[i] = np.zeros((batchsize, numdims), dtype='float32')
        data = self.data[i]
      datasize = 0  # Number of rows of data filled up.

      # First put any left overs from previous disk accesses.
      if self.left_overs[i] is not None:
        left_over_size = self.left_overs[i].shape[0]
        if left_over_size > batchsize:
          if sparse:
            data = self.left_overs[i][:batchsize]
          else:
            data[:batchsize] = self.left_overs[i][:batchsize]
          self.left_overs[i] = self.left_overs[i][batchsize:]
          datasize = batchsize
        else:
          if sparse:
            data = self.left_overs[i]
          else:
            data[:left_over_size] = self.left_overs[i]
          self.left_overs[i] = None
          datasize = left_over_size

      # Read data from disk.
      while(datasize < batchsize):
        if self.last_read_file[i] != current_file:
          this_chunk = self.ReadDiskData(filename_list[current_file], key)
          self.last_read_chunk[i] = this_chunk
          self.last_read_file[i] = current_file
        else:
          this_chunk = self.last_read_chunk[i]
        this_chunk_size = this_chunk.shape[0]

        if datasize + this_chunk_size > batchsize:
          # Put part of this_chunk into the data and remaining in left_overs.
          self.left_overs[i] = this_chunk[batchsize - datasize:]
          if sparse:
            data = self.AddSparseData(data, this_chunk[:batchsize - datasize])
          else:
            data[datasize : batchsize] = this_chunk[:batchsize - datasize]
          datasize = batchsize
        else:
          # Put whole of this_chunk into the data.
          self.left_overs[i] = None
          if sparse:
            data = self.AddSparseData(data, this_chunk)
          else:
            data[datasize : datasize + this_chunk_size] = this_chunk
          datasize += this_chunk_size
        current_file = (current_file + 1) % num_files
      data_list.append(data)
    return data_list

  @staticmethod
  def LoadPickle(inputfile, key=None, verbose=False):
    """Loads a pickle."""
    fo = gzip.GzipFile(inputfile, 'rb')
    spec = pickle.load(fo)
    if key:
      spec = spec[key].T
    fo.close()
    return spec

  @staticmethod
  def LoadSparse(inputfile, verbose=False):
    """Loads a sparse matrix stored as npz file."""
    npzfile = np.load(inputfile)
    mat = sp.csr_matrix((npzfile['data'], npzfile['indices'],
                                  npzfile['indptr']),
                                  shape=tuple(list(npzfile['shape'])))
    if verbose:
      print 'Loaded sparse matrix from %s of shape %s' % (inputfile,
                                                          mat.shape.__str__())
    return mat

  @staticmethod
  def SaveSparse(outputfile, mat, verbose=False):
    if verbose:
      print 'Saving to %s shape %s' % (outputfile, mat.shape.__str__())
    np.savez(outputfile, data=mat.data, indices=mat.indices, indptr=mat.indptr,
             shape=np.array(list(mat.shape)))



  def ReadDiskData(self, filename, key=''):
    """Reads data from filename."""
    if self.verbose:
      print 'Reading from disk %s' % filename
    ext = os.path.splitext(filename)[1]
    if ext == '.npy':
      data = np.load(filename)
    elif ext == '.mat':
      data = scipy.io.loadmat(filename, struct_as_record = True)[key]
    elif ext == '.p':
      data = pickle.load(gzip.GzipFile(filename, 'rb'))
    elif ext == '.txt':
      data = np.loadtext(filename)
    elif ext == '.npz':
      data = Disk.LoadSparse(filename, verbose=self.verbose)
    elif ext == '.spec':
      data = Disk.LoadPickle(filename, key, verbose=self.verbose)
    else:
      raise Exception('Unknown file extension %s' % ext)
    if data.dtype == 'float64':
      data = data.astype('float32')

    # 1-D data as column vector.
    if len(data.shape) == 1 or (len(data.shape)==2 and data.shape[0] == 1):
      data = data.reshape(-1, 1)

    return data


class Cache(object):
  def __init__(self, parent, capacity, numdim_list, typesize=4, randomize=False,
              verbose=False, **kwargs):
    """Initialize a Cache.
    Args:
      parent: object that will provide data to this cache. Must have a Get().
      capacity: Maximum number of bytes that can fit in the cache.
      numdim_list: List of dimensions of the data.
      typesize: size (in bytes) of an atomic data entry.
      randomize: If True, shuffle the vectors after receiving them from the
        parent.
      verbose: If True, print info about what is happening.
    """
    self.parent = parent
    self.num_data = len(numdim_list)
    self.numdims = sum(numdim_list)
    self.numdim_list = numdim_list
    self._maxpos = capacity / (self.numdims * typesize)
    self.verbose = verbose
    self.capacity = self._maxpos * self.numdims * typesize
    self.typesize = typesize
    self._pos = 0
    self.data = []
    self.datasize = 0
    self.randomize = randomize
    if self.verbose:
      print 'Capacity %d bytes for data of size %d X %d rand=%s' % (
        self.capacity, self._maxpos, self.numdims, randomize)

  def LoadData(self):
    """Load data from the parent."""

    # If cache has no data or it holds less data than parent, then it is
    # time to ask for more data.
    if self.data == [] or self._maxpos < self.parent._maxpos:
      self.data = self.parent.Get(self._maxpos)
      self.datasize = self.data[0].shape[0]

    if self.randomize:
      # Shuffle the data. Need to make sure same shuffle is applied to all data
      # pieces in the list.
      rng_state = np.random.get_state()
      for i, d in enumerate(self.data):
        if sp.issparse(d):  # Not easy to do in-place shuffling for sparse data.
          indices = np.arange(d.shape[0])
          np.random.set_state(rng_state)
          np.random.shuffle(indices)
          self.data[i] = d[indices]
        else:
          np.random.set_state(rng_state)
          np.random.shuffle(d)

  def Get(self, batchsize):
    """Get data points from the cache.
    Args:
      batchsize: Number of data points requested. Will return fewer than
        batchsize iff the cache does not have enough data.
    Returns:
      Numpy array slice of shape batchsize X numdims.
    """
    if self._pos == self.datasize:
      self._pos = 0
    if self._pos == 0:
      self.LoadData()
    start = self._pos
    end = self._pos + batchsize
    if end > self.datasize:
      end = self.datasize
    self._pos = end
    batch = [d[start:end] for d in self.data]
    return batch


class GPUCache(Cache):
  """GPU memory manager."""
  def __init__(self, *args, **kwargs):
    super(GPUCache, self).__init__(*args, **kwargs)
    
    self.data = [None] * self.num_data
    self.empty = True
    self.allocated_memory_size = [0] * self.num_data
    
    # Elementary preprocessing can be done on the GPU.
    self.normalize = [False] * self.num_data
    self.means = [None] * self.num_data
    self.stds = [None] * self.num_data

    # Add gaussian noise.
    self.add_noise = kwargs.get('add_noise', [False]*self.num_data)
    sigma = 0.01

    # Add random translations (useful for vision data).
    self.translate = kwargs.get('shift', [False]*self.num_data)
    shift_amt_x = kwargs.get('shift_amt_x', [0])[0]
    shift_amt_y = kwargs.get('shift_amt_y', [0])[0]
    center_only = kwargs.get('center_only', False)
    shift_amt = max(shift_amt_x, shift_amt_y)
    self.sizeX = 32  # Should pass this as arguments!
    self.sizex = 32 - 2 * shift_amt
    self.num_channels = 3
    if center_only:  # True for test data.
      self.translate_range_x = [0]
      self.translate_range_y = [0]
      self.sigma = 0
    else:
      self.translate_range_x = range(-shift_amt_x, shift_amt_x + 1)
      self.translate_range_y = range(-shift_amt_y, shift_amt_y + 1)
      self.sigma = sigma

    self.translated_d = None
    self.offset_x = None
    self.offset_y = None

  def Normalize(self):
    """Normalize the data present in self.data"""
    for i, batch in enumerate(self.data):
      if self.normalize[i]:
        mean = self.means[i]
        std = self.stds[i]
        batch.add_col_mult(mean, mult=-1.0)
        batch.div_by_col(std)

  def LoadData(self):
    """Load data from parent cache."""

    # Ask parent for data.
    data_cpu = self.parent.Get(self._maxpos)
    datasize = data_cpu[0].shape[0]
    assert datasize <= self._maxpos,\
      "GPU cache can only store %d datapoints, but parent gave it %d." % (
        self._maxpos, datasize)

    self.datasize = datasize
    for i, d in enumerate(data_cpu):
      if sp.issparse(d):
        mat = d.toarray().T
      else:
        mat = d.T
      size = mat.shape[0] * mat.shape[1]
      if size > self.allocated_memory_size[i]:
        # If need more space, then allocate new matrix on the GPU.
        self.data[i] = cm.CUDAMatrix(mat)
        self.allocated_memory_size[i] = mat.shape[0] * mat.shape[1]
      else:
        # Overwrite old memory. It is ok if size of mat is less than the total
        # space that has been allocated.
        self.data[i].overwrite(mat)
    self.Normalize()

  def AddNoise(self, batch, i):
    # Add gaussian noise to data at index i in batch.
    batch[i].sample_gaussian(mult=self.sigma)

  def TranslateData(self, batch, i):
    """Applies translations to data at index i in batch."""
    sizeX = self.sizeX
    sizex = self.sizex
    batchsize = batch[i].shape[1]
    shift = (sizeX - sizex)/2
    offset_x = np.array([random.choice(self.translate_range_x) + shift for k in range(batchsize)]).reshape(1, -1)
    offset_y = np.array([random.choice(self.translate_range_y) + shift for k in range(batchsize)]).reshape(1, -1)
    num_channels = self.num_channels

    d = batch[i]

    if self.offset_x is None:
      self.offset_x = cm.CUDAMatrix(offset_x)
    else:
      self.offset_x.overwrite(offset_x)
    if self.offset_y is None:
      self.offset_y = cm.CUDAMatrix(offset_y)
    else:
      self.offset_y.overwrite(offset_y)
    if self.translated_d is None or self.translated_d.shape[1] != batchsize:
      self.translated_d = cm.empty((sizex**2 * num_channels, batchsize))
    d.generate_translations(sizeX, sizex, self.offset_x, self.offset_y, target=self.translated_d)
    batch[i] = self.translated_d

  def ShuffleData(self):
    """In-place shuffle the data in self.data."""
    indices = np.arange(self.datasize)
    np.random.shuffle(indices)
    indices1 = indices[:self.datasize/2]
    indices2 = indices[self.datasize/2:2*(self.datasize/2)]
    indices1_gpu = cm.CUDAMatrix(indices1.reshape(1, -1))
    indices2_gpu = cm.CUDAMatrix(indices2.reshape(1, -1))
    for d in self.data:
      d.swap_columns(indices1_gpu, indices2_gpu, target=d)
    indices1_gpu.free_device_memory()
    indices2_gpu.free_device_memory()

  def SetDataStats(self, i, stats_file):
    """Load stats for normalizing the data."""
    assert os.path.exists(stats_file), 'Stats file %s not found.' % stats_file
    stats = np.load(stats_file)
    self.normalize[i] = True
    self.means[i] = cm.CUDAMatrix(stats['mean'].reshape(-1, 1))
    self.stds[i] = cm.CUDAMatrix(1e-10 + stats['std'].reshape(-1, 1))

  def Get(self, batchsize, get_last_piece=False):
    """Return 'batchsize' data points from the cache.
    
    May return fewer points towards the end of the dataset when there are fewer
    than batchsize left.
    """
    skip = False
    if self._pos == self.datasize:
      self._pos = 0
    if self._pos == 0:
      if self.empty or self._maxpos < self.parent._maxpos:
        self.LoadData()
        self.empty = False
      if self.randomize and self._maxpos == self.parent._maxpos:
        # Shuffle if randomize is True and parent has not already shuffled it.
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
      batch = [d.slice(start, end) for d in self.data]
      for i in range(self.num_data):
        if self.add_noise[i]:
          self.AddNoise(batch, i)
        if self.translate[i]:
          self.TranslateData(batch, i)
      return batch

def GetBytes(mem_str):
  """Converts human-readable numbers to bytes.

  E.g., converts '2.1M' to 2.1 * 1024 * 1024 bytes.
  """
  unit = mem_str[-1]
  val = float(mem_str[:-1])
  if unit == 'G':
    val *= 1024*1024*1024
  elif unit == 'M':
    val *= 1024*1024
  elif unit == 'K':
    val *= 1024
  else:
    try:
      val = int(mem_str)
    except Exception:
      print '%s is not a valid way of writing memory size.' % mem_str
  return int(val)

def GetDataHandles(op, names, hyp_list, verbose=False):
  """Returns a list of data handles.

  This method is the top-level routine for creating data handlers. It takes a
  description of which datasets to load and returns data handlers to access
  them.
  Args:
    op: Operation protocol buffer.
    names: list of list of data names. The top level list corresponds to train,
      validation and test sets. The lower-level lists correspond to data
      modalities.
    hyp_list: List of hyperparameters for each modality.
    verbose: If True, will print out details of what is happening.
  Returns:
    A list of DataHandler objects.
  """
  typesize = 4
  data_proto_file = os.path.join(op.data_proto_prefix, op.data_proto)
  dataset_proto = util.ReadData(data_proto_file)
  handlers = []
  if dataset_proto.data_handler == 'deepnet':
    size_list = []
    for name_list in names:
      size = 0
      for name in name_list:
        try:
          data_proto = next(d for d in dataset_proto.data if d.name == name)
        except StopIteration as e:
          print '%s not found in data pbtxt' % name
          raise e
        datasetsize = data_proto.size
        numdims = np.prod(np.array(data_proto.dimensions))
        size += datasetsize * numdims * typesize
      size_list.append(size)
    total_size = sum(size_list)
    proportions = [float(size)/total_size for size in size_list]
    for i, name_list in enumerate(names):
      if name_list == []:
        handlers.append(None)
      else:
        handlers.append(DataHandler(op, name_list, hyp_list, frac=proportions[i]))
  elif dataset_proto.data_handler == 'navdeep':
    import navdeep_datahandler
    for i, name_list in enumerate(names):
      if name_list == []:
        handlers.append(None)
      else:
        handlers.append(navdeep_datahandler.NavdeepDataHandler(
          op, dataset_proto, name_list, hyp_list))

  return handlers

class DataHandler(object):
  """Data handling class."""
  def __init__(self, op, data_name_list, hyperparameter_list, frac=1.0):
    """Initializes a DataHandler.
    Args:
      op: Operation protocol buffer.
      data_name_list: List of data names that should be put together. (Usually
        refers to a list of different modalities, e.g., ['data', 'label'] or
        ['image', 'audio'].)
      hyperparameter_list: List of hyperparameters, one for each modality.
      frac: What fraction of the total memory should this data handler use.
    """
    filenames = []
    numdim_list = []
    datasetsize = None
    left_window = []
    right_window = []
    stats_files = []
    shift = []
    add_noise = []
    shift_amt_x = []
    shift_amt_y = []
    keys = []
    typesize = 4
    if isinstance(op, str):
      op = util.ReadOperation(op)
    self.verbose = op.verbose
    verbose = self.verbose
    data_proto_file = os.path.join(op.data_proto_prefix, op.data_proto)
    dataset_proto = util.ReadData(data_proto_file)
    seq = False
    is_train = False
    for name, hyp in zip(data_name_list, hyperparameter_list):
      data_proto = next(d for d in dataset_proto.data if d.name == name)
      file_pattern = os.path.join(dataset_proto.prefix, data_proto.file_pattern)
      filenames.append(sorted(glob.glob(file_pattern)))
      stats_files.append(os.path.join(dataset_proto.prefix, data_proto.stats_file))
      numdims = np.prod(np.array(data_proto.dimensions))
      if not data_proto.sparse:
        numdims *= data_proto.num_labels
      numdim_list.append(numdims)
      seq = seq or data_proto.seq
      left_window.append(hyp.left_window)
      right_window.append(hyp.right_window)
      add_noise.append(hyp.add_noise)
      shift.append(hyp.shift)
      shift_amt_x.append(hyp.shift_amt_x)
      shift_amt_y.append(hyp.shift_amt_y)
      keys.append(data_proto.key)
      is_train = 'train' in name  # HACK - Fix this!
      if datasetsize is None:
        datasetsize = data_proto.size
      else:
        assert datasetsize == data_proto.size, 'Size of %s is not %d' % (
          name, datasetsize)

    # Add space for padding.
    if seq:
      max_rw = max(right_window)
      max_lw = max(left_window)
      actual_datasetsize = datasetsize
      datasetsize += len(filenames[0]) * (max_rw + max_lw)

    numdims = sum(numdim_list)
    batchsize = op.batchsize
    randomize = op.randomize
    self.get_last_piece = op.get_last_piece
    # Compute size of each cache.
    total_disk_space = datasetsize * numdims * typesize
    max_gpu_capacity = int(frac*GetBytes(dataset_proto.gpu_memory))
    max_cpu_capacity = int(frac*GetBytes(dataset_proto.main_memory))

    # Each capacity should correspond to integral number of batches.
    vectorsize_bytes = typesize * numdims
    batchsize_bytes = vectorsize_bytes * batchsize
    max_gpu_capacity = (max_gpu_capacity / batchsize_bytes) * batchsize_bytes
    #max_cpu_capacity = (max_cpu_capacity / batchsize_bytes) * batchsize_bytes

    # Don't need more than total dataset size.
    gpu_capacity = min(total_disk_space, max_gpu_capacity) 
    cpu_capacity = min(total_disk_space, max_cpu_capacity) 
    num_gpu_batches = gpu_capacity / batchsize_bytes
    num_cpu_batches = cpu_capacity / batchsize_bytes

    gpu_left_overs = gpu_capacity / vectorsize_bytes - num_gpu_batches * batchsize
    cpu_left_overs = cpu_capacity / vectorsize_bytes - num_cpu_batches * batchsize
    
    if self.verbose:
      if seq:
        num_valid_gpu_vectors = (gpu_capacity/vectorsize_bytes) - len(filenames[0])*(max_rw+max_lw)
        print num_valid_gpu_vectors

      else:
        print 'Batches in GPU memory: %d + leftovers %d' % (num_gpu_batches,
                                                            gpu_left_overs)
        print 'Batches in main memory: %d + leftovers %d' % (num_cpu_batches,
                                                             cpu_left_overs)
        print 'Batches in disk: %d + leftovers %d' % ((datasetsize / batchsize),
                                                      datasetsize % batchsize)
    
    if seq:
      import sequence_datahandler as seq_dh
      self.disk = seq_dh.SequenceDisk(
        filenames, numdim_list, datasetsize, keys=keys, left_window=left_window,
        right_window=right_window, verbose=verbose)
      self.cpu_cache = seq_dh.SequenceCache(
        self.disk, cpu_capacity, numdim_list, typesize = typesize,
        randomize=randomize, left_window=left_window,
        right_window=right_window, verbose=verbose)
      self.gpu_cache = seq_dh.SequenceGPUCache(
        self.cpu_cache, gpu_capacity, numdim_list, typesize = typesize,
        randomize=randomize, left_window=left_window,
        right_window=right_window, verbose=verbose, batchsize=batchsize)
    else:
      self.disk = Disk(filenames, numdim_list, datasetsize, keys=keys,
                       verbose=self.verbose)
      self.cpu_cache = Cache(self.disk, cpu_capacity, numdim_list,
                             typesize = typesize, randomize=randomize,
                             verbose=self.verbose)
      self.gpu_cache = GPUCache(self.cpu_cache, gpu_capacity, numdim_list,
                                typesize = typesize, randomize=randomize,
                                verbose=self.verbose, shift=shift, add_noise=add_noise,
                                center_only=not is_train, shift_amt_x=shift_amt_x, shift_amt_y=shift_amt_y)
    for i, stats_file in enumerate(stats_files):
      if hyperparameter_list[i].normalize and hyperparameter_list[i].activation != deepnet_pb2.Hyperparams.REPLICATED_SOFTMAX:
        self.gpu_cache.SetDataStats(i, stats_file)
    self.batchsize = batchsize
    if seq:
      datasetsize = actual_datasetsize
    self.num_batches = datasetsize / batchsize
    if self.get_last_piece and datasetsize % batchsize > 0:
      self.num_batches += 1

  def Get(self):
    """Returns a list of minibatches on the GPU.
    Each element of the list corresponds to one modality.
    """
    batch = self.gpu_cache.Get(self.batchsize, get_last_piece=self.get_last_piece)
    return batch

  def GetCPUBatches(self):
    """Returns batches from main memory."""
    batch = self.cpu_cache.Get(self.batchsize)
    return batch


class DataWriter(object):
  """Class for writing lots of data to disk."""

  def __init__(self, names, output_dir, memory, numdim_list, datasize=None):
    """Initializes a Data Writer.
    Args:
      names: Names used to identify the different data components. Will be used
        as prefixes for the output files.
      output_dir: Directory where the data will be written.
      memory: Size of each output chunk.
      numdim_list: Number of dimensions in each data component.
      datasize: Total number of data vectors that will be written. Having this
        number helps to save memory.
    """
    typesize = 4  # Fixed for now.
    self.typesize = typesize
    self.names = names
    self.output_dir = output_dir
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    self.numdim_list = numdim_list
    self.data_len = len(names)
    assert self.data_len == len(numdim_list)
    numdims = sum(numdim_list)
    total_memory = GetBytes(memory)
    if datasize is not None:
      total_memory_needed = datasize * typesize * numdims
      total_memory = min(total_memory, total_memory_needed)
    self.buffer_index = [0] * self.data_len
    self.dump_count = [0] * self.data_len
    self.data_written = [0] * self.data_len
    self.max_dumps = []
    self.buffers = []
    for numdim in numdim_list:
      memory = (total_memory * numdim) / numdims
      numvecs = memory / (typesize * numdim)
      data = np.zeros((numvecs, numdim), dtype='float32')
      self.buffers.append(data)
      if datasize is not None:
        max_dump = datasize / numvecs
        if datasize % numvecs > 0:
          max_dump += 1
        self.max_dumps.append(max_dump)
      else:
        self.max_dumps.append(1)

  def AddToBuffer(self, i, data):
    """Add data into buffer i."""
    buf = self.buffers[i]
    buf_index = self.buffer_index[i]
    datasize = data.shape[0]
    assert datasize + buf_index <= buf.shape[0], 'Not enough space in buffer.'
    buf[buf_index:buf_index + datasize] = data
    self.buffer_index[i] += datasize

  def FreeSpace(self, i):
    """Return amount of free space left."""
    return self.buffers[i].shape[0] - self.buffer_index[i]

  def HasSpace(self, i, datasize):
    """Return True if buffer i has space to add datasize more vectors."""
    buf = self.buffers[i]
    buf_index = self.buffer_index[i]
    return buf.shape[0] > buf_index + datasize
  
  def IsFull(self, i):
    return not self.HasSpace(i, 0)

  def DumpBuffer(self, i):
    """Write the contents of buffer i to disk."""
    buf_index = self.buffer_index[i]
    if buf_index == 0:
      return
    buf = self.buffers[i]
    output_prefix = os.path.join(self.output_dir, self.names[i])
    output_filename = '%s-%.5d-of-%.5d' % (
      output_prefix, (self.dump_count[i]+1), self.max_dumps[i])
    self.dump_count[i] += 1
    np.save(output_filename, buf[:buf_index])
    self.buffer_index[i] = 0
    self.data_written[i] += buf_index

  def SubmitOne(self, i, d):
    datasize = d.shape[0]
    free_space = self.FreeSpace(i)
    if datasize > free_space:
      self.AddToBuffer(i, d[:free_space])
    else:
      self.AddToBuffer(i, d)
    if self.IsFull(i):
      self.DumpBuffer(i)
    if datasize > free_space:
      self.SubmitOne(i, d[free_space:])

  def Submit(self, data):
    assert len(data) == self.data_len
    for i, d in enumerate(data):
      self.SubmitOne(i, d)

  def Commit(self):
    for i in range(self.data_len):
      self.DumpBuffer(i)
    return self.data_written
