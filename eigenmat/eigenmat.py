import os, pdb, platform, time, warnings
import ctypes as ct
import numpy as np

if platform.system() == 'Windows':
  _eigenmat = ct.cdll.LoadLibrary('libeigenmat.dll')
elif platform.system() == 'Darwin':
  _eigenmat = ct.cdll.LoadLibrary('libeigenmat.dylib')
else:
  _eigenmat = ct.cdll.LoadLibrary('libeigenmat.so')

_eigenmat.euclid_norm.restype = ct.c_float
_eigenmat.vdot.restype = ct.c_float
_eigenmat.sum_all.restype = ct.c_float

def deprecated(func):
  """This is a decorator which can be used to mark functions
  as deprecated. It will result in a warning being emmitted
  when the function is used."""

  def newFunc(*args, **kwargs):
    warnings.warn("Call to deprecated function %s." % func.__name__,
           category=DeprecationWarning)
    return func(*args, **kwargs)
  newFunc.__name__ = func.__name__
  newFunc.__doc__ = func.__doc__
  newFunc.__dict__.update(func.__dict__)
  return newFunc

class EigenMatException(Exception):
  pass

def get_last_cuda_error():
  return str(_eigenmat.get_last_cuda_error())

def generate_exception(err_code):
  """
  Return a EigenMatException object based on the error code err_code.
  """

  if err_code == -1:
    return EigenMatException("Incompatible matrix dimensions.")
  elif err_code == -2:
    return EigenMatException("CUBLAS error.")
  elif err_code == -3:
    return EigenMatException("CUDA error: " + get_last_cuda_error())
  elif err_code == -4:
    return EigenMatException("Operation not supported on views.")
  elif err_code == -5:
    return EigenMatException("Operation not supported on transposed matrices.")
  elif err_code == -6:
    return EigenMatException("")
  elif err_code == -7:
    return EigenMatException("Incompatible transposedness.")
  elif err_code == -8:
    return EigenMatException("Matrix is not in device memory.")
  elif err_code == -9:
    return EigenMatException("Operation not supported.")
    

class eigenmat(ct.Structure):
  _fields_ = [('data', ct.POINTER(ct.c_float)),
        ('size', ct.c_int * 2),
        ('is_trans', ct.c_int),
        ('owns_data', ct.c_int)]

class rnd_struct(ct.Structure):
  _fields_ = [('seed', ct.c_ulong),
        ('kn', ct.c_int * 128),
        ('fn', ct.c_float * 128),
        ('wn', ct.c_float * 128)]


class TransposedEigenMatrix(object):
  def __init__(self, mat):
    self.mat = eigenmat()
    ct.memmove(ct.pointer(self.mat), ct.pointer(mat), ct.sizeof(self.mat))
    self.mat.is_trans = 1
    self.p_mat = ct.pointer(self.mat)
    self.T = mat

class EigenMatrix(object):
  """
  A EigenMatrix object represents a matrix of single precision floating point
  numbers on a GPU.
  """

  def overwrite(self, array):
    """Overwrites self with array.
    
    'array' should have a size smaller than that of the array used to
    initialize the EigenMatrix. The method will not throw an Exception just
    yet if this is not true. It will throw exceptions or behave in strange
    ways later on.
    """
    assert type(array) == np.ndarray, 'array must be a np.ndarray.'
    array = reformat(array)
    self.numpy_array = array
    _eigenmat.init_from_array(self.p_mat, array.ctypes.data_as(ct.POINTER(ct.c_float)), ct.c_int(array.shape[0]), ct.c_int(array.shape[1]))

  def __init__(self, array, **kwargs):
    """
    Initializes a new matrix object in one of two ways. If array is a numpy
    ndarray, memory for a matrix with the same dimensions is allocated on
    the GPU. If the copy_to_device flag is set to True, the GPU matrix is
    initialized with the given ndarray. If array is not an ndarray, it must
    be a eigenmat structure (typically the user will never use this way of
    calling __init__).
    """

    if type(array) == np.ndarray:
      # Convert array to float32 in FORTRAN order
      array = reformat(array)

      # Initialize as a ndarray-tied matrix.
      self.mat = eigenmat()
      self.size = self.mat.size
      self.p_mat = ct.pointer(self.mat)
      self.numpy_array = array

      _eigenmat.init_from_array(self.p_mat, array.ctypes.data_as(ct.POINTER(ct.c_float)), ct.c_int(array.shape[0]), ct.c_int(array.shape[1]))
    else:
      # Initialize based on existing eigenmat structure.
      self.mat = array
      self.p_mat = ct.pointer(self.mat)
    self.T = TransposedEigenMatrix(self.mat)

  @staticmethod
  def init_random(seed=0):
    """
    Initialize and seed the random number generator.
    """
    assert seed >= 0, "Seed must be a non-negative integer."
    EigenMatrix.rnd_state = rnd_struct()
    EigenMatrix.rnd_state_p = ct.pointer(EigenMatrix.rnd_state)
    _eigenmat.init_random(EigenMatrix.rnd_state_p, ct.c_int(seed+1))
 

  @property
  def shape(self):
    return (self.mat.size[0], self.mat.size[1])

  def set_shape(self, shape):
    """
    Sets the shape of the array to the given array.
    Highly unsafe method. Does no checking.
    Do not use this unless you know what you are doing.
    """

    m = ct.c_uint(shape[0])
    n = ct.c_uint(shape[1])

    err_code = _eigenmat.set_shape(self.p_mat, m, n)
    if err_code:
      raise generate_exception(err_code)

    return self

  def reshape(self, shape):
    """
    Reshapes self to have the given shape. The number of elements cannot
    change as this only changes how the contents are interpreted.
    """

    m = ct.c_uint(shape[0])
    n = ct.c_uint(shape[1])

    err_code = _eigenmat.reshape(self.p_mat, m, n)
    if err_code:
      raise generate_exception(err_code)

    return self

  def blockify(source, blocksize, target=None):
    if target == None:
      target = source

    err_code = _eigenmat.blockify(source.p_mat, target.p_mat, ct.c_uint(blocksize))

    if err_code:
      raise generate_exception(err_code)

    return target

  def generate_translations(source, source_w, target_w, off_x, off_y, target=None):
    num_channels = source.shape[0] / (source_w**2)

    if target == None:
      batch_s = source.shape[1]
      target = empty((target_w**2, batch_s))

    err_code = _eigenmat.generate_translations_big_var_off(source.p_mat, target.p_mat, off_x.p_mat, off_y.p_mat, ct.c_uint(source_w), ct.c_uint(target_w), ct.c_uint(num_channels))

    if err_code:
      raise generate_exception(err_code)

    return target

  def asarray(self):
    """
    Copies the matrix to an ndarray on the CPU and returns it.
    """
    return self.numpy_array

  def copy_to_device(self):
    """
    Copy the matrix to the GPU.
    """
    pass

  def copy_to_host(self):
    """
    Copy the matrix to the CPU.
    """
    pass

  def assign(self, val):
    """Assign val to self, where val can be a scalar or a EigenMatrix
    with the same dimensions as self. """

    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.copy_on_device(val.p_mat, self.p_mat)
    elif isinstance(val, (int, float)):
      err_code = _eigenmat.assign_scalar(self.p_mat, ct.c_float(val))
    else:
      raise ValueError, "Assigned value must be of type EigenMatrix, int, or float."
      
    if err_code:
      raise generate_exception(err_code)

    return self

  def free_device_memory(self):
    """
    Free memory used up by the matrix on the GPU.
    """
    pass

  def set_trans(self, is_trans):
    """
    Set the transposedness flag to is_trans.
    """
    _eigenmat.set_transpose(self.p_mat, ct.c_int(1 * is_trans))

  def slice(self, first_col, last_col):
    mat = eigenmat()

    if self.mat.size[0] == 1 or self.mat.size[1] == 1:
      err_code = _eigenmat.get_vector_slice(self.p_mat, ct.pointer(mat), ct.c_int(first_col), ct.c_int(last_col))
    else:
      err_code = _eigenmat.get_slice(self.p_mat, ct.pointer(mat), ct.c_int(first_col), ct.c_int(last_col))

    if err_code:
      raise generate_exception(err_code)

    new_mat = EigenMatrix(mat)

    try:
      new_mat.sliceof = self.sliceof
    except:
      new_mat.sliceof = self

    return new_mat

  def get_col_slice(self, first_col, last_col, target=None):
    col_slice = self.slice(first_col, last_col)

    if target:
      target.assign(col_slice)
      return target
    else:
      return col_slice

  def set_col_slice(self, first_col, last_col, mat):
    self.slice(first_col, last_col).assign(mat)

    return self

  def get_row_slice(self, start, end, target=None):
    """
    Get the rows with indices start through end. If target is not provided
    memory for a new matrix will be allocated.
    """

    width = self.shape[1]

    if not target:
      target = empty((end-start, width))

    err_code = _eigenmat.get_row_slice(self.p_mat, target.p_mat, ct.c_int(start), ct.c_int(end))
    if err_code:
      raise generate_exception(err_code)

    return target

  def set_row_slice(self, start, end, mat):
    """
    Assign the contents of mat to the rows with indices start through end.
    """

    err_code = _eigenmat.set_row_slice(mat.p_mat, self.p_mat, ct.c_int(start), ct.c_int(end))
    if err_code:
      raise generate_exception(err_code)

    return self

  def transpose(self, target=None):
    """
    Return a transposed copy of the matrix.
    """
    if not target:
      target = empty((self.shape[1], self.shape[0]))

    err_code = _eigenmat.copy_transpose(self.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return target

  def fill_with_rand(self):
    """
    Fill matrix on the GPU with random numbers drawn from the uniform
    distribution over the (0,1) interval.
    """

    err_code = _eigenmat.fill_with_rand(EigenMatrix.rnd_state_p, self.p_mat) 
    if err_code:
      raise generate_exception(err_code)

    return self

  def fill_with_randn(self):
    """
    Fill matrix on the GPU with random numbers drawn from the standard normal
    distribution.
    """

    err_code = _eigenmat.fill_with_randn(EigenMatrix.rnd_state_p, self.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return self

  def dropout(self, dropprob, val=0.0):
    """
    Drop entries in this matrix uniformly randomly with given probability
    and set the dropped out unit to state val.
    """
    err_code = _eigenmat.dropout(EigenMatrix.rnd_state_p, self.p_mat,
                  ct.c_float(dropprob), ct.c_float(val))
    if err_code:
      raise generate_exception(err_code)

    return self

  def sample_bernoulli(self, target=None):
    """
    Sample a bernoulli distribution. Choose 1 with probability given by entries of self, 0 otherwise.
    """
    if not target:
     target = self
    err_code = _eigenmat.sample_bernoulli(EigenMatrix.rnd_state_p, self.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return self

  def sample_bernoulli_tanh(self, target=None):
    """
    Sample a bernoulli distribution. Choose 1 with probability given by entries of (1+self)/2, -1 otherwise.
    """
    if not target:
     target = self
    err_code = _eigenmat.sample_bernoulli_tanh(EigenMatrix.rnd_state_p, self.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return self

  def sample_poisson(self, target=None):
    """
    Sample a poisson distribution. Choose 1 with probability given by entries of self.
    Not implemented yet.
    """
    if not target:
     target = self
    err_code = _eigenmat.sample_poisson(EigenMatrix.rnd_state_p, self.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return self

  def sample_gaussian(self, mult=1.0, target=None):
    """
    Add zero mean gaussian noise to the matrix. mult is the stddev.
    """
    if not target:
     target = self
    err_code = _eigenmat.sample_gaussian(EigenMatrix.rnd_state_p, self.p_mat, target.p_mat, ct.c_float(mult))
    if err_code:
      raise generate_exception(err_code)

    return self

  def perturb_energy_for_softmax_sampling(self, target=None):
    """
    Add by -log(-log(rand)).
    """
    if not target:
     target = self
    err_code = _eigenmat.perturb_energy(EigenMatrix.rnd_state_p, self.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return self

  def perturb_prob_for_softmax_sampling(self, target=None):
    """
    Divide by -log(rand).
    """
    if not target:
     target = self
    err_code = _eigenmat.perturb_prob(EigenMatrix.rnd_state_p, self.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return self


  def add_col_vec(self, vec, target=None):
    """
    Add vector vec to every column of the matrix. If a target is provided,
    it is used to store the result instead of self.
    """

    if not target:
      target = self

    err_code = _eigenmat.add_col_vec(self.p_mat, vec.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return target
    
  def add_col_mult(self, vec, mult, target=None):
    """
    Add a multiple of vector vec to every column of the matrix. If a target
    is provided, it is used to store the result instead of self.
    """

    if not target:
      target = self

    err_code = _eigenmat.add_col_mult(self.p_mat, vec.p_mat, target.p_mat, ct.c_float(mult))
    if err_code:
      raise generate_exception(err_code)

    return target

  def add_mult_sign(self, mat2, mult = 1.):
    """
    Add multiple of sign of mat2 to the matrix.
    """

    err_code = _eigenmat.add_mult_sign(self.p_mat, mat2.p_mat, ct.c_float(mult))
    if err_code:
      raise generate_exception(err_code)

    return self

  def mult_diagonal(self, val, target=None):
    """
    Mult val to the diagonal of self. If a target
    is provided, it is used to store the result instead of self.
    """

    if not target:
      target = self

    assert self.shape[0] == self.shape[1], 'self must be a square matrix'
    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.mult_diagonal(self.p_mat, val.p_mat, target.p_mat)
    elif isinstance(val, (int, float)):
      err_code = _eigenmat.mult_diagonal_scalar(self.p_mat, ct.c_float(val), target.p_mat)
    else:
      raise ValueError, "Value must be of type EigenMatrix, int, or float."

    if err_code:
      raise generate_exception(err_code)

    return target



  def add_diagonal(self, val, target=None):
    """
    Add val to the diagonal of self. If a target
    is provided, it is used to store the result instead of self.
    """

    if not target:
      target = self

    assert self.shape[0] == self.shape[1], 'self must be a square matrix'
    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.add_diagonal(self.p_mat, val.p_mat, target.p_mat)
    elif isinstance(val, (int, float)):
      err_code = _eigenmat.add_diagonal_scalar(self.p_mat, ct.c_float(val), target.p_mat)
    else:
      raise ValueError, "Value must be of type EigenMatrix, int, or float."

    if err_code:
      raise generate_exception(err_code)

    return target


  def add_row_mult(self, vec, mult, target=None):
    """
    Add a multiple of vector vec to every row of the matrix. If a target
    is provided, it is used to store the result instead of self.
    """

    if not target:
      target = self

    err_code = _eigenmat.add_row_mult(self.p_mat, vec.p_mat, target.p_mat, ct.c_float(mult))
    if err_code:
      raise generate_exception(err_code)

    return target
    
  def add_row_vec(self, vec, target=None):
    """
    Add vector vec to every row of the matrix. If a target is provided,
    it is used to store the result instead of self.
    """

    if not target:
      target = self

    err_code = _eigenmat.add_row_vec(self.p_mat, vec.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return target
    
  def mult_by_col(self, vec, target=None):
    """
    Multiply vector vec into every column of the matrix. If a target is
    provided, it is used to store the result instead of self.
    """

    if not target:
      target = self

    err_code = _eigenmat.mult_by_col_vec(self.p_mat, vec.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return target
    
  def mult_by_row(self, vec, target=None):
    """
    Multiply vector vec into every row of the matrix. If a target is
    provided, it is used to store the result instead of self.
    """

    if not target:
      target = self

    err_code = _eigenmat.mult_by_row_vec(self.p_mat, vec.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return target

  def div_by_col(self, vec, target=None):
    """
    Multiply vector vec into every column of the matrix. If a target is
    provided, it is used to store the result instead of self.
    """

    if not target:
      target = self

    err_code = _eigenmat.div_by_col_vec(self.p_mat, vec.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return target
    
  def div_by_row(self, vec, target=None):
    """
    Divide vector vec into every row of the matrix. If a target is
    provided, it is used to store the result instead of self.
    """

    if not target:
      target = self

    err_code = _eigenmat.div_by_row_vec(self.p_mat, vec.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return target
 
  def sum(self, axis=None, target = None):
    """
    Sum the matrix along the given dimension, where 0 represents the leading
    dimension and 1 represents the non-leading dimension. If None, the sum
    of all elements is returned. If a target is not prvided, a new vector is
    created for storing the result.
    """
    if axis is None:
     return _eigenmat.sum_all(self.p_mat)
    else:
     return sum(self, axis, target)


  def add_sums(self, mat, axis, mult = 1.):
    """
    Add a multiple of the sums of the matrix mat along the given dimension
    to self. 
    """

    m = _eigenmat.get_leading_dimension(mat.p_mat)
    n = _eigenmat.get_nonleading_dimension(mat.p_mat)

    err_code = _eigenmat.add_sum_by_axis(mat.p_mat, self.p_mat, ct.c_int(axis), ct.c_float(mult))
    if err_code:
      raise generate_exception(err_code)

    return self

  def less_than(self, val, target=None):
    """
    Perform the operation target = 1. * (self < val), where val can be a matrix or a scalar.
    """

    if not target:
      target = self

    if isinstance(val, (int, float)):
      err_code = _eigenmat.less_than_scalar(self.p_mat, ct.c_float(val), target.p_mat)
    else:
      err_code = _eigenmat.less_than(self.p_mat, val.p_mat, target.p_mat)

    if err_code:
      raise generate_exception(err_code)

    return target

  def greater_than(self, val, target=None):
    """
    Perform the operation target = 1. * (self > val), where val can be a matrix or a scalar.
    """

    if not target:
      target = self

    if isinstance(val, (int, float)):
      err_code = _eigenmat.greater_than_scalar(self.p_mat, ct.c_float(val), target.p_mat)
    else:
      err_code = _eigenmat.greater_than(self.p_mat, val.p_mat, target.p_mat)

    if err_code:
      raise generate_exception(err_code)

    return target

  def upper_bound(self, val, target=None):
    """
    Perform the operation target = (self > val) ? val:self, where val can be a matrix or a scalar.
    """
    if not target:
      target = self

    if isinstance(val, (int, float)):
      err_code = _eigenmat.upper_bound_scalar(self.p_mat, ct.c_float(val), target.p_mat)
    else:
      err_code = _eigenmat.upper_bound(self.p_mat, val.p_mat, target.p_mat)

    if err_code:
      raise generate_exception(err_code)

    return target


  def lower_bound(self, val, target=None):
    """
    Perform the operation target = (self < val) ? val:self, where val can be a matrix or a scalar.
    """
    if not target:
      target = self

    if isinstance(val, (int, float)):
      err_code = _eigenmat.lower_bound_scalar(self.p_mat, ct.c_float(val), target.p_mat)
    else:
      err_code = _eigenmat.lower_bound(self.p_mat, val.p_mat, target.p_mat)

    if err_code:
      raise generate_exception(err_code)

    return target

  def cumsum(self, axis, temp=None, target=None):
    """
    Cumulative sum along axis.
    """

    m, n = self.shape
    assert axis == 0, 'axis = 1 not implemented.'
    if not target:
      target = empty((m, n))
    if not temp:
      temp = empty((m, n))
    """ 
    elif axis == 1:
      if not target:
        target = empty((m, 1))
    """ 

    err_code = _eigenmat.cumsum_by_axis(self.p_mat, target.p_mat, temp.p_mat, ct.c_int(axis))
    if err_code:
      raise generate_exception(err_code)

    return target

  def choose_max_and_accumulate(self, acc):
    """
    Find the maximum value along the given dimension, where 0 represents the
    leading dimension and 1 represents the non-leading dimension. If a target
    is not prvided, a new vector is created for storing the result.
    """

    m, n = self.shape

    err_code = _eigenmat.choose_max_and_accumulate(self.p_mat, acc.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return acc


  def choose_max(self, axis, target=None):
    """
    Find the maximum value along the given dimension, where 0 represents the
    leading dimension and 1 represents the non-leading dimension. If a target
    is not prvided, a new vector is created for storing the result.
    """

    m, n = self.shape

    assert axis == 0, 'Axis = 1 not implemented.'
    if not target:
     target = self

    err_code = _eigenmat.choose_max_by_axis(self.p_mat, target.p_mat, ct.c_int(axis))
    if err_code:
      raise generate_exception(err_code)

    return target


  def max(self, axis, target=None):
    """
    Find the maximum value along the given dimension, where 0 represents the
    leading dimension and 1 represents the non-leading dimension. If a target
    is not prvided, a new vector is created for storing the result.
    """

    m, n = self.shape

    if axis == 0:
      if not target:
        target = empty((1, n))
 
    elif axis == 1:
      if not target:
        target = empty((m, 1))

    err_code = _eigenmat.max_by_axis(self.p_mat, target.p_mat, ct.c_int(axis))
    if err_code:
      raise generate_exception(err_code)

    return target

  def argmax(self, axis, target=None):
    """
    Find the index with the maximum value along the given dimension, where 0 represents the
    leading dimension and 1 represents the non-leading dimension. If a target
    is not prvided, a new vector is created for storing the result.
    """

    m, n = self.shape

    if axis == 0:
      if not target:
        target = empty((1, n))
 
    elif axis == 1:
      if not target:
        target = empty((m, 1))

    err_code = _eigenmat.argmax_by_axis(self.p_mat, target.p_mat, ct.c_int(axis))
    if err_code:
      raise generate_exception(err_code)

    return target


  def sqsum(self, axis, target=None):
    """
    Find the sum of squares along the given dimension, where 0 represents the
    leading dimension and 1 represents the non-leading dimension. If a target
    is not prvided, a new vector is created for storing the result.
    """

    m, n = self.shape

    if axis == 0:
      if not target:
        target = empty((1, n))
 
    elif axis == 1:
      if not target:
        target = empty((m, 1))

    err_code = _eigenmat.sqsum_by_axis(self.p_mat, target.p_mat, ct.c_int(axis))
    if err_code:
      raise generate_exception(err_code)

    return target

  def norm_limit(self, norm, axis, target=None):
    """
    Limit the norm along the given dimension to be 'norm', where 0
    represents the leading dimension and 1 represents the non-leading
    dimension. If a target is not provided, self is used as target.
    """
    m, n = self.shape

    if axis == 0:
      if not target:
        target = self
 
    elif axis == 1:
      if not target:
        target = self

    err_code = _eigenmat.normlimit_by_axis(self.p_mat, target.p_mat,
                        ct.c_int(axis), ct.c_float(norm))
    if err_code:
      raise generate_exception(err_code)

    return target

  def sign(self, target=None):
    """
    Find the sign of each element of the matrix.
    """

    if not target:
      target = empty((self.mat.size[0], self.mat.size[1]))

    err_code = _eigenmat.sign(self.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return target

  def apply_cos(self, target=None):
    """
    Apply the cos sigmoid to each element of the matrix.
    """

    return cos(self, target)

  def apply_sin(self, target=None):
    """
    Apply the sin sigmoid to each element of the matrix.
    """

    return sin(self, target)

  def apply_sigmoid(self, target=None):
    """
    Apply the logistic sigmoid to each element of the matrix.
    """

    return sigmoid(self, target)

  def apply_softmax(self, target=None):
    """
    Apply softmax activation. Each column is taken as one softmax.
    """
    return softmax(self, target)


  def reciprocal(self, target=None):
    """
    Find the reciprocal of each element of the matrix.
    """

    if not target:
      target = self

    err_code = _eigenmat.reciprocal(self.p_mat, target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return target

  def dot(self, mat2, mult=1.0, target=None):
    """
    Multiply the matrix by mat2 from the right and multiply by scalar mult.
    """

    return dot(self, mat2, mult, target)

  def add_dot(self, m1, m2, mult=1.0):
    """
    Add the dot product of m1 and m2 to the matrix.
    """

    err_code = _eigenmat.dot(m1.p_mat, m2.p_mat, self.p_mat, ct.c_float(1.), ct.c_float(mult))
    if err_code:
      raise generate_exception(err_code)

    return self

  def subtract_dot(self, m1, m2):
    """
    Subtract the dot product of m1 and m2 from the matrix.
    """

    err_code = _eigenmat.dot(m1.p_mat, m2.p_mat, self.p_mat, ct.c_float(1.), ct.c_float(-1.))
    if err_code:
      raise generate_exception(err_code)

    return self

  def add_mult(self, mat2, alpha = 1.):
    """
    Add multiple of mat2 to the matrix.
    """

    err_code = _eigenmat.add_mult(self.p_mat, mat2.p_mat, ct.c_float(alpha))
    if err_code:
      raise generate_exception(err_code)

    return self
  
  def subtract_mult(self, mat2, alpha = 1.):
    """
    Subtract a multiple of mat2 from the matrix.
    """

    err_code = _eigenmat.add_mult(self.p_mat, mat2.p_mat, ct.c_float(-1. * alpha))
    if err_code:
      raise generate_exception(err_code)

    return self

  def add(self, val, target=None):
    """Add val to self, where val can be a scalar or a EigenMatrix with the
    same dimensions as self. """

    if not target:
      target = self

    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.add_elementwise(self.p_mat, val.p_mat, target.p_mat)
    elif isinstance(val, (int, float)):
      err_code = _eigenmat.add_scalar(self.p_mat, ct.c_float(val), target.p_mat)
    else:
      raise ValueError, "Value must be of type EigenMatrix, int, or float."

    if err_code:
      raise generate_exception(err_code)

    return target

  def subtract(self, val, target=None):
    """Subtract val from self, where val can be a scalar or a EigenMatrix with
    the same dimensions as self. """

    if not target:
      target = self

    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.subtract_elementwise(self.p_mat, val.p_mat, target.p_mat)
    elif isinstance(val, (int, float)):
      err_code = _eigenmat.add_scalar(self.p_mat, ct.c_float(-1*val), target.p_mat)
    else:
      raise ValueError, "Value must be of type EigenMatrix, int, or float."

    if err_code:
      raise generate_exception(err_code)

    return target

  def divide(self, val, target=None):
    """Divide self by val, where val can be a scalar or a EigenMatrix with the
    same dimensions as self. """

    if not target:
      target = self

    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.divide_elementwise(self.p_mat, val.p_mat, target.p_mat)
    elif isinstance(val, (int, float)):
      err_code = _eigenmat.divide_by_scalar(self.p_mat, ct.c_float(val), target.p_mat)
    else:
      raise ValueError, "Value must be of type EigenMatrix, int, or float."

    if err_code:
      raise generate_exception(err_code)

    return target

  def mult(self, val, target=None):
    """Multiply self by val, where val can be a scalar or a EigenMatrix with
    the same dimensions as self. """

    if not target:
      target = self

    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.mult_elementwise(self.p_mat, val.p_mat, target.p_mat)
    elif isinstance(val, (int, float)):
      err_code = _eigenmat.mult_by_scalar(self.p_mat, ct.c_float(val), target.p_mat)
    else:
      raise ValueError, "Value must be of type EigenMatrix, int, or float."

    if err_code:
      raise generate_exception(err_code)

    return target

  def apply_cos_deriv(self, val, target=None):
    """
    Apply cos derivative, where val is the activation of cos units.
    """

    if not target:
      target = self

    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.apply_cos_deriv(self.p_mat, val.p_mat, target.p_mat)
    else:
      raise ValueError, "Value must be of type EigenMatrix."

    if err_code:
      raise generate_exception(err_code)

    return target


  def apply_sin_deriv(self, val, target=None):
    """
    Apply sin derivative, where val is the activation of sin units.
    """

    if not target:
      target = self

    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.apply_sin_deriv(self.p_mat, val.p_mat, target.p_mat)
    else:
      raise ValueError, "Value must be of type EigenMatrix."

    if err_code:
      raise generate_exception(err_code)

    return target


  def apply_logistic_deriv(self, val, target=None):
    """
    Apply logistic derivative, where val is the activation of logistic units.
    """

    if not target:
      target = self

    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.apply_logistic_deriv(self.p_mat, val.p_mat, target.p_mat)
    else:
      raise ValueError, "Value must be of type EigenMatrix."

    if err_code:
      raise generate_exception(err_code)

    return target

  def apply_tanh_deriv(self, val, target=None):
    """
    Apply tanh derivative, where val is the activation of the units.
    """

    if not target:
      target = self

    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.apply_tanh_deriv(self.p_mat, val.p_mat, target.p_mat)
    else:
      raise ValueError, "Value must be of type EigenMatrix."

    if err_code:
      raise generate_exception(err_code)

    return target

  def apply_rectified_linear_deriv(self, val, target=None):
    """
    Apply rectified linear derivative, where val is the activation of the units.
    """

    if not target:
      target = self

    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.apply_rectified_linear_deriv(self.p_mat, val.p_mat, target.p_mat)
    else:
      raise ValueError, "Value must be of type EigenMatrix."

    if err_code:
      raise generate_exception(err_code)

    return target

  def apply_rectified_linear_smooth_deriv(self, val, target=None):
    """
    Apply rectified linear smooth derivative, where val is the activation of the units.
    """

    if not target:
      target = self

    if isinstance(val, EigenMatrix):
      err_code = _eigenmat.apply_rectified_linear_smooth_deriv(self.p_mat, val.p_mat, target.p_mat)
    else:
      raise ValueError, "Value must be of type EigenMatrix."

    if err_code:
      raise generate_exception(err_code)

    return target

  @deprecated
  def assign_scalar(self, alpha):
    """
    Assign scalar alpha to every element of the matrix.
    """

    err_code = _eigenmat.assign_scalar(self.p_mat, ct.c_float(alpha))
    if err_code:
      raise generate_exception(err_code)

    return self

  @deprecated
  def mult_by_scalar(self, alpha, target=None):
    """
    Multiply the matrix by a scalar.
    """

    if not target:
      target = self

    err_code = _eigenmat.mult_by_scalar(self.p_mat, ct.c_float(alpha), target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return target


  @deprecated
  def div_by_scalar(self, alpha, target=None):
    """
    Divide the matrix by a scalar.
    """

    if not target:
      target = self

    err_code = _eigenmat.divide_by_scalar(self.p_mat, ct.c_float(alpha), target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return target

  @deprecated
  def add_scalar(self, alpha, target=None):
    """
    Increment the matrix by a scalar.
    """

    if not target:
      target = self

    err_code = _eigenmat.add_scalar(self.p_mat, ct.c_float(alpha), target.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return target

  def sum_all(self):
    err_code = ct.c_int(0)
    res = _eigenmat.sum_all(self.p_mat)

    if err_code:
      raise generate_exception(err_code.value, ct.byref(err_code))

    return res

  def euclid_norm(self):
    err_code = ct.c_int(0)
    res = _eigenmat.euclid_norm(self.p_mat, ct.byref(err_code))

    if err_code:
      raise generate_exception(err_code.value)

    return res

  def select_columns(self, indices, target):
    """
    copies some columns of self into target.
    <indices> must be a row vector. Its elements are float32's representing integers, e.g. "34.0" means the integer "34".
    after this call, for all r,c, target[r,c]=self[r,indices[c]].
    This returns target.
    Negative indices are interpreted in the usual Python way: all elements of <indices> had better be in the range [-self.shape[1], self.shape[1]-1].
    This does bounds checking, but out of bounds indices do not raise an exception (because the programmer was lazy). Instead, they result in NaN values in <target>.
    """

    err_code = _eigenmat.selectRows(self.p_mat, target.p_mat, indices.p_mat)

    if err_code:
      raise generate_exception(err_code)

    return target


  def swap_columns(self, indices1, indices2, target):
    """
    swap columns at indices1 of self with columns at indices2 of target.
    <indices1> and <indices2> must be row vectors of equal length. Its elements are float32's representing integers, e.g. "34.0" means the integer "34".
    after this call, for all r,c, target[r,indices2[c]=self[r,indices1[c]].
    self can be same as target, but then the result will be non-deterministic if there is overlap between indices1 and indices2. Can be used for in-place shuffling by making sure indices1 and indices2 do not overlap.
    This returns target.
    Negative indices are interpreted in the usual Python way: all elements of <indices> had better be in the range [-self.shape[1], self.shape[1]-1].
    This does bounds checking, but out of bounds indices do not raise an exception (because the programmer was lazy). Instead, they result in NaN values in <target>.
    """
    assert indices1.shape[0] == 1
    assert indices1.shape == indices2.shape
    err_code = _eigenmat.swapCols(self.p_mat, target.p_mat, indices1.p_mat, indices2.p_mat)

    if err_code:
      raise generate_exception(err_code)

    return target

  def set_selected_columns(self, indices, source):
    """
    copies all columns of source into some columns of self.
    <indices> must be a row vector. Its elements are float32's representing
    integers, e.g. "34.0" means the integer "34". after this call, for all
    r,c, self[r,indices[c]]=source[r,c]. This returns self.
    Negative indices are interpreted in the usual Python way: all elements
    of <indices> had better be in the range [-self.shape[1], self.shape[1]-1].
    This does bounds checking, but out of bounds indices do not raise an
    exception (because the programmer was lazy). Instead, they result in NaN
    values in <self>.
    """

    err_code = _eigenmat.setSelectedRows(self.p_mat, source.p_mat, indices.p_mat)
    if err_code:
      raise generate_exception(err_code)

    return self

  def get_softmax_correct(self, labels, target):
    """
    target[i] = 1, iff labels[i] is correctly predicted; 0 otherwise.
    """
    assert labels.shape == (1, self.shape[1])
    assert target.shape == labels.shape
    if isinstance(labels, EigenMatrix):
      err_code = _eigenmat.get_softmax_correct(self.p_mat, labels.p_mat, target.p_mat)
    else:
      raise ValueError, "labels must be of type CUDAMatrix."

    if err_code:
      raise generate_exception(err_code)

    return target

  def get_softmax_cross_entropy(self, labels, target, tiny=1e-10):
    """
    target[i] = -log(self[label[i]] + tiny).
    """
    assert labels.shape == (1, self.shape[1])
    assert target.shape == labels.shape
    if isinstance(labels, EigenMatrix):
      err_code = _eigenmat.get_softmax_cross_entropy(self.p_mat, labels.p_mat, target.p_mat, ct.c_float(tiny))
    else:
      raise ValueError, "labels must be of type EigenMatrix or CUDAMatrix."

    if err_code:
      raise generate_exception(err_code)

    return target

  def apply_softmax_grad(self, labels, target = None):
    """
    Apply softmax derivative, where labels are the correct labels.
    """
    if not target:
      target = self

    assert labels.shape == (1, self.shape[1])
    assert target.shape == self.shape
    if isinstance(labels, EigenMatrix):
      err_code = _eigenmat.apply_softmax_grad(self.p_mat, labels.p_mat, target.p_mat)
    else:
      raise ValueError, "labels must be of type EigenMatrix or CUDAMatrix."

    if err_code:
      raise generate_exception(err_code)

    return target




CUDAMatrix = EigenMatrix
def empty(shape):
  """
  Creates and returns a new EigenMatrix with the given shape.
  """
  return EigenMatrix(np.zeros(shape))

def sum(mat, axis, target=None):
  """
  Sum the matrix along the given dimension, where 0 represents the leading
  dimension and 1 represents the non-leading dimension. If a target is
  not prvided, a new vector is created for storing the result.
  """

  m = _eigenmat.get_leading_dimension(mat.p_mat)
  n = _eigenmat.get_nonleading_dimension(mat.p_mat)

  if axis == 0:
    # sum along leading dimension
    if not target:
      target = empty((1, n))
 
  elif axis == 1:
    # sum along non-leading dimension
    if not target:
      target = empty((m, 1))

  err_code = _eigenmat.sum_by_axis(mat.p_mat, target.p_mat, ct.c_int(axis))

  if err_code:
    raise generate_exception(err_code)

  return target

def dot(m1, m2, mult=1.0, target=None):
  """
  Find the dot product between m1 and m2.
  """

  if not target:
    m = _eigenmat.get_leading_dimension(m1.p_mat)
    n = _eigenmat.get_nonleading_dimension(m2.p_mat)

    target = empty((m, n))

  err_code = _eigenmat.dot(m1.p_mat, m2.p_mat, target.p_mat, ct.c_float(0.), ct.c_float(mult))
  if err_code:
    raise generate_exception(err_code)

  return target

def vdot(m1, m2):
  """
  Compute the vector dot product of matrices m1 and m2.
  """

  err_code = ct.c_int(0)
  res = _eigenmat.vdot(m1.p_mat, m2.p_mat, ct.byref(err_code))

  if err_code:
    raise generate_exception(err_code.value)

  return res

def cos(mat, target=None):
  """
  Apply cos to each element of the matrix mat.
  """

  if not target:
    target = mat

  err_code = _eigenmat.apply_cos(mat.p_mat, target.p_mat)
  if err_code:
    raise generate_exception(err_code)

  return target



def sin(mat, target=None):
  """
  Apply sin to each element of the matrix mat.
  """

  if not target:
    target = mat

  err_code = _eigenmat.apply_sin(mat.p_mat, target.p_mat)
  if err_code:
    raise generate_exception(err_code)

  return target

def softmax(mat, target = None):
  """
  Apply softmax activation to each column of mat.
  """
  if not target:
    target = mat

  err_code = _eigenmat.apply_softmax(mat.p_mat, target.p_mat)
  if err_code:
    raise generate_exception(err_code)
  return target

def sigmoid(mat, target=None):
  """
  Apply the logistic sigmoid to each element of the matrix mat.
  """

  if not target:
    target = mat

  err_code = _eigenmat.apply_sigmoid(mat.p_mat, target.p_mat)
  if err_code:
    raise generate_exception(err_code)

  return target

def tanh(mat, target=None):
  """
  Apply the tanh to each element of the matrix mat.
  """

  if not target:
    target = mat

  err_code = _eigenmat.apply_tanh(mat.p_mat, target.p_mat)
  if err_code:
    raise generate_exception(err_code)

  return target

def abs(mat, target=None):
  """
  Apply abs to each element of the matrix mat.
  """

  if not target:
    target = mat

  err_code = _eigenmat.apply_abs(mat.p_mat, target.p_mat)
  if err_code:
    raise generate_exception(err_code)

  return target

def log_1_plus_exp(mat, target=None, exact=False):
  """
  Apply log(1+exp(x)) to each element of the matrix mat. If exact is True, use
  slow and accurate log and exp.
  """

  if not target:
    target = mat

  if exact:
   err_code = _eigenmat.apply_log_1_plus_exp_exact(mat.p_mat, target.p_mat)
  else:
   err_code = _eigenmat.apply_log_1_plus_exp(mat.p_mat, target.p_mat)
  if err_code:
    raise generate_exception(err_code)

  return target

def log(mat, tiny=0.0, target=None):
  """
  Find the natural logarithm of each element of the matrix mat.
  """

  if not target:
    target = mat

  err_code = _eigenmat.apply_log(mat.p_mat, target.p_mat, ct.c_float(tiny))
  if err_code:
    raise generate_exception(err_code)

  return target

def exp(mat, target=None):
  """
  Apply the exponential function to each element of the matrix mat.
  """

  if not target:
    target = mat

  err_code = _eigenmat.apply_exp(mat.p_mat, target.p_mat)
  if err_code:
    raise generate_exception(err_code)

  return target

def ceil(mat, target=None):
  """
  Apply the ceil function to each element of the matrix mat.
  """

  if not target:
    target = mat

  err_code = _eigenmat.apply_ceil(mat.p_mat, target.p_mat)
  if err_code:
    raise generate_exception(err_code)

  return target

def floor(mat, target=None):
  """
  Apply the floor function to each element of the matrix mat.
  """

  if not target:
    target = mat

  err_code = _eigenmat.apply_floor(mat.p_mat, target.p_mat)
  if err_code:
    raise generate_exception(err_code)

  return target

def sqrt(mat, target=None):
  """
  Compute the square root of each element of the matrix mat.
  """

  if not target:
    target = mat

  err_code = _eigenmat.apply_sqrt(mat.p_mat, target.p_mat)
  if err_code:
    raise generate_exception(err_code)

  return target

def cross_entropy_bernoulli(mat, p, target=None, tiny=1e-10):
  """
  Compute -mat*log(p) - (1-mat).*log(1-p)
  """

  if not target:
    target = mat

  if isinstance(p, EigenMatrix):
    err_code = _eigenmat.compute_cross_entropy_bernoulli(mat.p_mat, p.p_mat, target.p_mat, ct.c_float(tiny))
  else:
    raise ValueError, "Value must be of type EigenMatrix."

  if err_code:
    raise generate_exception(err_code)

  return target


def cross_entropy(mat, p, target=None, tiny=1e-10):
  """
  Compute -mat*log(p)
  """

  if not target:
    target = mat

  if isinstance(p, EigenMatrix):
    err_code = _eigenmat.compute_cross_entropy(mat.p_mat, p.p_mat, target.p_mat, ct.c_float(tiny))
  else:
    raise ValueError, "Value must be of type EigenMatrix."

  if err_code:
    raise generate_exception(err_code)

  return target

def correct_preds(mat, p, target=None, cutoff=0.5):
  """
  Compute mat*(p >= 0.5) + (1-mat).*(p < 0.5)
  """

  if not target:
    target = mat

  if isinstance(p, EigenMatrix):
    err_code = _eigenmat.correct_preds(mat.p_mat, p.p_mat, target.p_mat, ct.c_float(cutoff))
  else:
    raise ValueError, "Value must be of type EigenMatrix."

  if err_code:
    raise generate_exception(err_code)

  return target

def pow(mat, p, target=None):
  """
  If p is a scalar, compute the 'p'th power of each element of the matrix mat,
  otherwise raise each element of the matrix mat to the power given by the
  corresponding element of the matrix p.
  """

  if not target:
    target = mat

  if isinstance(p, EigenMatrix):
    err_code = _eigenmat.apply_pow_matrix(mat.p_mat, p.p_mat, target.p_mat)
  elif isinstance(p, (int, float)):
    err_code = _eigenmat.apply_pow(mat.p_mat, ct.c_float(p), target.p_mat)
  else:
    raise ValueError, "Value must be of type EigenMatrix, int, or float."

  if err_code:
    raise generate_exception(err_code)

  return target

def cuda_sync_threads():
  pass

def reformat(array):
  """
  Returns array as a float32 array in FORTRAN order.
  """
  return np.array(array, dtype=np.float32, order='F')

def cuda_set_device(dev_id):
  """
  Selects the CUDA device with the given ID.
  """
  pass

def cublas_init():
  """
  Initialize Cublas.
  """
  pass

init = cublas_init

def cublas_shutdown():
  """
  Shut down Cublas.
  """
  pass

shutdown = cublas_shutdown
