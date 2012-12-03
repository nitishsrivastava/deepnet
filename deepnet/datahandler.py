"""Data handling class."""

import cPickle as pickle
import cudamat as cm
import deepnet_pb2
import glob
import numpy as np
import os.path
import scipy.sparse

def DataHandle(*args, **kwargs):
  proto = args[1]
  data_format = proto.data_format
  if data_format == "protocolbuffer":
    if proto.seq:
      DH = SequentialDataHandler
    elif proto.sparse:
      DH = SparseDataHandler
    else:
      DH = DataHandler
  elif data_format == "navdeep_data":
    import navdeep_datahandler
    DH = navdeep_datahandler.NavdeepDataHandler
  else:
    raise Exception('Unknown data format.')
  return DH(*args, **kwargs)

class Cache(object):
  def __init__(self, name, source, capacity, parent_capacity, blocksize=1,
               typesize=4, gpu=False, verbose=False, numdims=1,
               randomize=False, permutation_link=None):
    """Simulates a cache.
    Args:
      name: Name of this cache.
      source: The place where this cache gets the data from
              (usually another Cache object).
      capacity: Capacity of the cache in bytes.
      blocksize: Number of elements that will be read in one atomic read.
      typesize: sizeof(float), usually 4.
      gpu: Is the cache on the gpu?
        Will store the data in cudamat array for gpu, numpy for main memory.
    """
    self.name = name
    self.source = source
    self.randomize = randomize
    self.capacity = capacity / typesize
    self.current_size = 0
    self.blocksize = blocksize
    self.typesize = typesize
    self.gpu = gpu
    self.numdims = numdims
    self.num_blocks = parent_capacity / capacity
    self.permutation_link = permutation_link
    if (parent_capacity % capacity) > 0: 
      self.num_blocks += 1
    self.Clear()
    self.verbose = verbose
    self.permutation_link = permutation_link
    if permutation_link:
      self.indices = self.permutation_link.indices
    else:
      self.indices = None
    if verbose:
      print '%s Num blocks: %d' % (self.name, self.num_blocks)
      print 'Capacity: %d' % self.capacity
      print 'Blocksize: %d' % self.blocksize

  def Clear(self):
    self._data = None
    self._position = 0

  def TupleMultiply(self, t):
    return reduce(lambda a, x: a * x, list(t))

  def Get(self, blocksize=None):
    if blocksize and blocksize != self.blocksize:
      self.blocksize = blocksize
    if self._position >= self.current_size or self._data is None:
      self._position = 0
      if self.num_blocks > 1 or self._data is None:
        if self.verbose:
          print 'CACHE MISS in %s' % self.name
        if self.gpu:
          if self._data:
            self._data.free_device_memory()
          self._data = cm.CUDAMatrix(
            self.source.Get((self.capacity)).reshape(1, -1))
          self.current_size = self._data.shape[1]
        else:
          self._data = self.source.Get((self.capacity))
          self.current_size = self._data.shape[0]
      if self.randomize:
        if self.permutation_link:
          self.indices = self.permutation_link.indices
        else:
          p = np.arange(self.current_size / self.numdims)
          np.random.shuffle(p)
          if self.gpu:
            if self.indices is not None:
              self.indices.free_device_memory()
            p2 = p.view()
            p2.shape = 1, -1
            self.indices = cm.CUDAMatrix(p2)
          else:
            self.indices = p
        if self.gpu:
          self._data.reshape((self.numdims, self.current_size / self.numdims))
          shuffled_data = cm.empty(self._data.shape)
          self._data.select_columns(self.indices, target=shuffled_data)
          self._data.free_device_memory()
          self._data = shuffled_data
          self._data.reshape((1, self.current_size))
        else:
          view = self._data.view()
          view.shape = self.current_size / self.numdims, self.numdims
          self._data = view[self.indices,:].reshape(-1)

    span = min(self.blocksize, self.current_size - self._position)
    self._position += span
    if self.gpu:
      return self._data.slice(self._position - span, self._position)
    else:
      return self._data[self._position - span : self._position]

class Disk(Cache):

  def __init__(self, name, filenames, data_size, blocksize,
               typesize=4, verbose=False, numdims=1):
    self.name = name
    self.current_file = 0
    self.numdims = numdims
    self.filenames = filenames
    self.blocksize = blocksize
    self.num_blocks = data_size / (typesize * blocksize)
    self.max_blocksize = data_size / typesize
    if self.num_blocks == 0:
      self.blocksize = self.max_blocksize
      self.num_blocks = 1
    elif (data_size % (typesize * blocksize)) > 0:
      self.num_blocks += 1
    self.num_files = len(self.filenames)
    self.gpu = False
    self._data = None
    self.verbose = verbose
    self.mean = None
    self.subtract_mean = False
    self.stddev = None
    self.divide_stddev = False

  def ComputeDataStats(self, write_proto=False):
    data = self.Get(flatten=False)
    assert data.dtype == 'float32'
    self.mean = data.mean(axis=0)
    self.stddev = data.std(axis=0) + np.exp(-30)
    assert self.mean.dtype == 'float32'

  def SetDataStats(self, proto, subtract_mean=False, divide_stddev=False):
    if not proto.mean_centered and subtract_mean:
      print 'Subtracting mean'
      self.mean = np.fromstring(proto.mean, dtype='float32')
      self.subtract_mean = True
    if not proto.unit_variance and divide_stddev:
      print 'Dividing by stddev'
      self.stddev = np.fromstring(proto.stddev, dtype='float32')
      self.divide_stddev = True


  def ReadDiskData(self, filename):
    if self.verbose:
      print 'Reading from disk %s' % filename
    ext = os.path.splitext(filename)[1]
    if ext == '.npy':
      data = np.load(filename)
    elif ext == '.mat':
      if 'key' in kwargs.keys():
          key = kwargs['key']
      else:
          key = desc
      data = scipy.io.loadmat(filename, struct_as_record = True)[key]
    elif ext == '.p':
      data = pickle.load(gzip.GzipFile(filename, 'rb'))
    elif ext == '.txt':
      data = np.loadtext(filename)
    elif ext == '.npz':
      data = self.load_sparse(filename)
    else:
      raise Exception('Unknown file extension %s' % ext)
    if data.dtype == 'float64':
      data = data.astype('float32')
    if self.subtract_mean:
      data -= self.mean
    if self.divide_stddev:
      data /= self.stddev
    if len(data.shape) == 1 or (len(data.shape)==2 and (
      data.shape[0] == 1 or data.shape[1] == 1)):
      data = data.reshape(-1, 1)
    return data

  def Get(self, blocksize=None, flatten=True):
    if self.num_blocks > 1 or self._data is None:
      if not blocksize:
        blocksize = self.blocksize
      if blocksize > self.max_blocksize:
        blocksize = self.max_blocksize
      total_datasize = blocksize / self.numdims
      data = np.zeros((total_datasize, self.numdims), dtype='float32')
      datasize = 0
      while(datasize < total_datasize):
        this_chunk = self.ReadDiskData(self.filenames[self.current_file])
        this_chunk_size, numdims = this_chunk.shape
        assert numdims == self.numdims, "Disk %s was told data has numdims %d, "\
            "but loaded data had numdims %d" % (self.name, self.numdims, numdims)
        if datasize + this_chunk_size > total_datasize:
          break
        data[datasize:datasize+this_chunk_size] = this_chunk
        datasize += this_chunk_size
        self.current_file = (self.current_file + 1) % self.num_files
      self._data = data[:datasize,:]
      if flatten:
        self._data.shape = (-1)
    return self._data

def AlignUp(a, b):
  c = a / b
  if a % b > 0:
    c += 1
  return c * b


class DataHandler(object):

  def __init__(self, name, proto, op, hyp, typesize=4,
               boundary_proto=None, permutation_link=None):
    self.name = name
    batchsize = op.batchsize
    randomize = op.randomize
    skip_last_piece = op.skip_last_piece
    filenames = sorted(glob.glob(proto.file_pattern))
    self.num_batches = proto.size / batchsize
    if not skip_last_piece and proto.size % batchsize > 0:
      self.num_batches += 1
    numdims = reduce(lambda a, x: a * x, proto.dimensions)
    datasetsize = proto.size
    disk_mem = datasetsize * typesize * numdims
    gpu_blocksize = numdims * batchsize
    gpu_mem = self.GetBytes(proto.gpu_memory)
    gpu_mem = min(gpu_mem, disk_mem)
    gpu_mem = AlignUp(gpu_mem, typesize * gpu_blocksize)
    
    main_mem = self.GetBytes(proto.main_memory)
    main_mem = min(main_mem, disk_mem)
    if main_mem < gpu_mem:
      main_mem = gpu_mem
    main_blocksize = gpu_mem / typesize
    
    disk_blocksize = main_mem / typesize

    disk = Disk(name+'_disk', filenames, disk_mem, disk_blocksize, numdims=numdims)
    disk.SetDataStats(proto, hyp.subtract_mean, hyp.divide_stddev)
    if permutation_link:
      link = permutation_link.main_memory_cache
      _randomize = permutation_link.main_memory_cache.randomize
    else:
      link = None
      _randomize = randomize
    self.main_memory_cache = Cache(name+'_main_mem_cache', disk, main_mem,
                                   disk_mem, main_blocksize, numdims=numdims,
                                   randomize=_randomize, permutation_link=link)
    if permutation_link:
      link = permutation_link.gpu_cache
      _randomize = permutation_link.gpu_cache.randomize
    else:
      link = None
      _randomize = randomize and main_mem / gpu_mem == 1
      # If main_mem is bigger then shuffling will take place on the cpu.
    self.gpu_cache = Cache(name+'_gpu_cache', self.main_memory_cache, gpu_mem,
                           main_mem, gpu_blocksize, gpu=True, numdims=numdims,
                           randomize=_randomize, permutation_link=link)
    
    self.shape = list(tuple(proto.dimensions))
    self.shape.append(batchsize)
    self.numdims = numdims
    self.batchsize = batchsize
    self.skip_last_piece = skip_last_piece

  def GetBytes(self, mem_str):
    unit = mem_str[-1]
    val = int(mem_str[:-1])
    if unit == 'G':
      val *= 1024*1024*1024
    elif unit == 'M':
      val *= 1024*1024
    elif unit == 'K':
      val *= 1024
    return val

  def Get(self):
    """Return a batch after reshaping appropriately."""
    s = self.gpu_cache.Get()
    batchsize = s.shape[1] / self.numdims
    if batchsize != self.batchsize and self.skip_last_piece:
      return self.Get()
    self.shape[-1] = batchsize
    s.reshape(tuple(self.shape))
    return s

class SequentialDataHandler(DataHandler):

  def __init__(self, name, proto, op, hyp, typesize=4, boundary_proto=None,
               permutation_link=None):
    self.name = name
    batchsize = op.batchsize
    skip_last_piece = op.skip_last_piece
    if permutation_link:
      left = 0
      right = 0
    else:
      left = hyp.left_window
      right = hyp.right_window
    randomize = op.randomize
    filenames = sorted(glob.glob(proto.file_pattern))
    self.num_batches = proto.size / batchsize
    if not skip_last_piece and proto.size % batchsize > 0:
      self.num_batches += 1
    numdims = reduce(lambda a, x: a * x, proto.dimensions)
    datasetsize = proto.size
    disk_mem = datasetsize * typesize * numdims
    gpu_blocksize = numdims * batchsize
    gpu_mem = self.GetBytes(proto.gpu_memory)
    gpu_mem = min(gpu_mem, disk_mem)
    gpu_mem = AlignUp(gpu_mem, typesize * gpu_blocksize)
    
    main_mem = self.GetBytes(proto.main_memory)
    main_mem = min(main_mem, disk_mem)
    if main_mem < gpu_mem:
      main_mem = gpu_mem
    main_blocksize = gpu_mem / typesize
    
    disk_blocksize = main_mem / typesize

    disk = Disk(name+'_disk', filenames, disk_mem, disk_blocksize)
    main_memory_cache = Cache(name+'_main_mem_cache', disk, main_mem,
                              disk_mem, main_blocksize)

    if permutation_link:
      link = permutation_link.gpu_cache
    else:
      link = None
    self.gpu_cache = SequentialCache(name+'_gpu_cache',main_memory_cache,
                                     numdims, left, right,
                                     gpu_mem, main_mem, gpu_blocksize,
                                     gpu=True, randomize=randomize,
                                     permutation_link=link)
    self.shape = list(tuple(proto.dimensions))
    self.shape.append(batchsize)
    self.numdims = numdims
    self.batchsize = batchsize
    self.skip_last_piece = skip_last_piece

  def Get(self):
    """Return a batch after reshaping appropriately."""
    s = self.gpu_cache.Get()
    batchsize = s.shape[1]
    if batchsize != self.batchsize and self.skip_last_piece:
      return self.Get()
    return s

class SequentialCache(Cache):

  def __init__(self, name, source, numdims, left, right,
               capacity, parent_capacity, blocksize=1,
               typesize=4, gpu=True, randomize=False, permutation_link=None,
               verbose=False):
    assert gpu, 'Sequential cache only works for GPU.'
    super(SequentialCache, self).__init__(name, source, capacity,
                                          parent_capacity, blocksize=blocksize,
                                          typesize=typesize, gpu=True,
                                          verbose=verbose)

    batchsize = self.blocksize / numdims
    self.batchsize = batchsize
    self.numdims = numdims
    self.left = left
    self.right = right
    self.numframes = 1 + left + right
    self.position_template = cm.CUDAMatrix(np.tile(
      np.arange(-left, right + 1, 1.0).reshape(-1, 1), (1, batchsize)))
    self.positions = cm.CUDAMatrix(np.zeros((self.numframes, batchsize)))
    self.output = cm.CUDAMatrix(np.zeros((self.numframes * self.numdims,
                                          batchsize)))
    self.randomize = randomize
    self.permutation_link = permutation_link
    if permutation_link:
      self.indices = self.permutation_link.indices
      if self.verbose:
        print '%s -> %s' % (self.name, self.permutation_link.name)
    else:
      self.indices = cm.CUDAMatrix(np.zeros(batchsize).reshape(1, -1))
      if not randomize:
        self.indices_init = cm.CUDAMatrix(
          np.arange(batchsize).reshape(1, -1) - batchsize)

  def Get(self, batchsize=None):
    if batchsize and batchsize != self.batchsize:
      self.batchsize = batchsize
    if self._position >= self.current_size or self._data is None:
      self._position = 0
      if not self.permutation_link and not self.randomize:
        self.indices.assign(self.indices_init)
      if self.num_blocks > 1 or self._data is None:
        if self.verbose:
          print 'CACHE MISS in %s' % self.name
        if self._data:
          self._data.free_device_memory()
        self._data = cm.CUDAMatrix(
          self.source.Get((self.capacity)).reshape(1, -1))
        size = self._data.shape[1]
        self._data.reshape((self.numdims, size / self.numdims))
        self.current_size = self._data.shape[1]
    positions = self.positions
    position_template = self.position_template
    numframes = self.numframes
    output = self.output
    batchsize = self.batchsize
    numdims = self.numdims
    indices = self.indices
    if not self.permutation_link:
      if self.randomize:
        indices.fill_with_rand()
        indices.mult(self.current_size)
      else:
        indices.add(self.batchsize)
    self._position += self.batchsize
    last_piece = False
    if self._position > self.current_size:
      last_piece = True
      span = self.current_size - self._position + batchsize
    position_template.add_row_vec(indices, target=positions)
    positions.reshape((1, numframes * batchsize))

    output.reshape((numdims, numframes * batchsize))
    self._data.select_columns(positions, target=output)
    output.reshape((numdims * numframes, batchsize))
    positions.reshape((numframes, batchsize))
    if last_piece:
      return output.slice(0, span)
    return output


class SparseDataHandler(DataHandler):
  def __init__(self, name, proto, op, hyp, typesize=4,
               boundary_proto=None, permutation_link=None):
    assert proto.sparse, "This class is meant for sparse data sets only."
    self.name = name
    batchsize = op.batchsize
    randomize = op.randomize
    skip_last_piece = op.skip_last_piece
    filenames = sorted(glob.glob(proto.file_pattern))
    assert len(filenames) == 1, filenames
    filename = filenames[0]
    self.num_batches = proto.size / batchsize
    if not skip_last_piece and proto.size % batchsize > 0:
      self.num_batches += 1
    numdims = reduce(lambda a, x: a * x, proto.dimensions) * proto.num_labels
    datasetsize = proto.size

    gpu_blocksize = numdims * batchsize
    gpu_mem = self.GetBytes(proto.gpu_memory)
    gpu_mem = AlignUp(gpu_mem, typesize * gpu_blocksize)
   
    main_mem = datasetsize * numdims * typesize
    print datasetsize, numdims, main_mem, gpu_mem
    main_blocksize = gpu_mem / typesize

    if permutation_link:
      link = permutation_link.main_memory_cache
    else:
      link = None
    self.sparse_memory_cache = SparseCache(name + '_sparse_cache', filename,
                                           main_blocksize, numdims,
                                           randomize=randomize,
                                           permutation_link=link)
    if permutation_link:
      link = permutation_link.gpu_cache
    else:
      link = None

    _randomize = randomize and main_mem / gpu_mem == 1
    
    self.gpu_cache = Cache(name+'_gpu_cache', self.sparse_memory_cache,
                           gpu_mem, main_mem, gpu_blocksize, gpu=True,
                           numdims=numdims,
                           randomize=_randomize, permutation_link=link)
    self.shape = list(tuple(proto.dimensions))
    self.shape[-1] *= proto.num_labels
    print '%s %s' % (self.name, self.shape)
    self.shape.append(batchsize)
    self.numdims = numdims
    self.batchsize = batchsize
    self.skip_last_piece = skip_last_piece

class SparseCache(Cache):

  @staticmethod
  def LoadSparse(inputfile):
    print 'Reading from disk %s' % inputfile
    npzfile = np.load(inputfile)
    mat = scipy.sparse.csr_matrix((npzfile['data'], npzfile['indices'],
                                  npzfile['indptr']),
                                  shape=tuple(list(npzfile['shape'])))
    print 'Loaded sparse matrix from %s of shape %s' % (inputfile,
                                                        mat.shape.__str__())
    return mat

  def __init__(self, name, sparse_data_file, blocksize, numdims,
               randomize=False, permutation_link=None):
    self.name = name
    self.data = None
    self.sparse_data_file = sparse_data_file
    self.batchsize = blocksize / numdims
    self.randomize = randomize
    self.permutation_link = permutation_link

  def Shuffle(self):
    if self.permutation_link is None:
      self.indices = np.arange(self.data.shape[0])
      np.random.shuffle(self.indices)
      self.data = self.data[self.indices,:]
    else:
      self.data = self.data[self.permutation_link.indices,:]


  def Get(self, capacity):
    if self.data is None:
      self.data = SparseCache.LoadSparse(self.sparse_data_file)
      self._position = 0
      self.max_position = self.data.shape[0]
      if self.randomize:
        self.Shuffle()
    if self._position >= self.max_position:
      assert self._position == self.max_position
      self._position = 0
      if self.randomize:
        self.Shuffle()
    if self._position + self.batchsize > self.max_position:
      this_batchsize = self.max_position - self._position
    else:
      this_batchsize = self.batchsize
    d = self.data[self._position:self._position + this_batchsize, :].toarray()
    self._position += this_batchsize
    return d
