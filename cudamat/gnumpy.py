"""Documentation can be found at http://www.cs.toronto.edu/~tijmen/gnumpy.html"""

"""

Copyright (c) 2010-2011 Tijmen Tieleman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

If you use Gnumpy for scientific work that gets published, you should include
in that publication a citation of the technical report that describes Gnumpy.
That report can be found at http://www.cs.toronto.edu/~tijmen/gnumpyTr.pdf

"""





"""
This file is not intended to be read by anyone other than gnumpy developers. It's long, it's weakly documented (much of the internal documentation is elsewhere), and many lines are unnaturally long & illegible because I did a lot of inlining.

If you really want to know how gnumpy works internally, or if you want to extend it, you can ask me for the original, which doesn't have the inlining, and the internal documentation.
"""




# ------------------------------------------------------------------------------- module init & shutdown

import numpy, operator as op, sys as _sys, types as _types, time as _time, os as _os, __builtin__, collections as _collections, pdb as _pdb, gc as _gc

_useGpu = _os.environ.get('GNUMPY_USE_GPU', 'auto')
assert _useGpu in ('auto', 'yes', 'no'), "environment variable GNUMPY_USE_GPU, if present, should be one of 'auto', 'yes', 'no'."
if _useGpu == 'auto':
 try: import cudamat as _cudamat; _useGpu = 'yes'
 except: print 'gnumpy: failed to import cudamat. Using npmat instead. No GPU will be used.'; _useGpu = 'no'
if _useGpu == 'yes':
 import cudamat as _cudamat
elif _useGpu == 'no':
 import npmat as _cudamat
 _precision = _os.environ.get('GNUMPY_CPU_PRECISION', '32')
 assert _precision in ('32', '64', '128'), 'environment variable GNUMPY_CPU_PRECISION, if present, should have value 32, 64, or 128.'
 _cudamat.__DTYPE__ = eval('numpy.float'+_precision)

_cmType = _cudamat.CUDAMatrix

def board_id_to_use():
 try:
  import gpu_lock
  return gpu_lock.obtain_lock_id()
 except:
  print 'gnumpy: failed to use gpu_lock. Using board #0 without knowing whether it is in use or not.'
  return 0
 
_boardId = None
def _init_gpu():
 """ picks a board and claims it (if using cudamat aot npmat). exception if there is no board. """
 if '__gpu_inited' in globals(): return
 global _boardId
 if _useGpu=='yes':
  _boardId = ( board_id_to_use() if callable(board_id_to_use) else board_id_to_use)
  if _boardId==-1: raise Exception('No gpu board is available. gnumpy will not function. Consider telling it to run on the CPU by setting environment variable GNUMPY_USE_GPU to "no".')
  _cudamat.cuda_set_device(_boardId)
  _cudamat.cublas_init()
 _cudamat.CUDAMatrix.init_random(0)
 globals()['__gpu_inited'] = None

expensive_check_probability = 1 
acceptable_number_types = 'anything goes' # alternatives: 'no nans'; 'no nans or infs'
dont__check_number_types_in_non_garrays = True
class GnumpyNumberTypeException(Exception): pass

_checking_number_type_now = False
def _check_number_types(x):
 """ does some checks, and then returns x. """
 if acceptable_number_types == 'anything goes': return x # this is the typical case, and in this case I just want to leave this checking function asap.

 global _checking_number_type_now
 if dont__check_number_types_in_non_garrays and not isinstance(x, garray): return x
 if _checking_number_type_now: return x # to prevent checks upon checks upon checks (infinite recursion)
 try:
  _checking_number_type_now = True
  if acceptable_number_types == 'no nans': raise NotImplementedError
  elif acceptable_number_types == 'no nans or infs':
   if not garray(x, copy=False).all_real(): raise GnumpyNumberTypeException('Found values that violate the rule set by gnumpy.acceptable_number_types: "%s"' % acceptable_number_types)
  else: assert False, 'gnumpy: the value of variable "acceptable_number_types" must be one of "anything goes", "no nans", "no nans or infs".'
 finally:
  _checking_number_type_now = False
 return x  
 


# ------------------------------------------------------------------------------- helpers copied from other files

def _isFullSlice(x): return type(x) == _types.SliceType and x == slice(None) # the first check is necessary to avoid returning a broadcast array of False's if x is an array
def _isSequence(x): return type(x) == list or type(x) == tuple or type(x)==xrange
def _insertT(tup, index, tupleToInsert): return tuple(tup[:index]) + tuple(tupleToInsert) + tuple(tup[index:])
def _modifyT(tup, index, newValue): return tuple(tup[:index]) + (newValue,) + tuple(tup[index+1:])
def _deleteT(tup, start, end): return tup[:start] + tup[end:]
def _prodT(x): return reduce(op.mul, x, 1)
def _findIndex3(tupOrGenerator): return ( i for i, x in enumerate(tuple(tupOrGenerator)) if x).next()
def _isNumber(x): return type(x) in _numberTypes
def _nonSeqAsS(x): return ( x if _isSequence(x) else (x,))
_t0=()
def reduceAdd(x): return reduce(op.add, x)

def _deleteT2(tup, index):
 index %= len(tup)
 return tup[:index] + tup[index+1:]

_intTypes = set((_types.IntType, _types.LongType, numpy.int16, numpy.int32, numpy.int8, numpy.int64))
_floatTypes = set((_types.FloatType, numpy.float64, numpy.float32, getattr(numpy, 'float128', numpy.float64), getattr(numpy, 'float96', numpy.float64))) # considering numpy.float64 a number is debatable. it really is a numpy object, and behaves that way, too: it has a __mul__ which prevents garray.__rmul__ from getting the task. However, for most purposes it's a number.
_numberTypes = _intTypes | _floatTypes
 
def _allTheSame(tup):
 tup = tuple(tup)
 if len(tup)<=1: return True
 for elt in tup[1:]:
  if elt != tup[0]: return False
 return True





# ------------------------------------------------------------------------------- gnumpy specific helpers

def _all2_(t, pred): return reduce(op.and_, map(pred, t), True)
def _any2_(t, pred): return reduce(op.or_, map(pred, t), False)

def _doExpensiveCheck(): return numpy.random.rand() < expensive_check_probability

def as_garray(x): return ( x if isinstance(x, garray) else garray(x))
def as_garray_or_scalar(x): return ( x if type(x) in _numberTypes or isinstance(x, garray) else garray(x))
def as_numpy_array(x): return ( x.as_numpy_array() if isinstance(x, garray) else numpy.array(x))

def _cm_reshape(cm, newShape):
 if _prodT(newShape)==0: return cm
 else: return cm.reshape(tuple(reversed(newShape)))

def _cm_col_slice_write(cm, start, end, sourceCm):
 cm.set_row_slice(start, end, sourceCm)

def _cm_col_slice_read(cm, start, end, target):
 cm.get_row_slice(start, end, target)
 return target

def _cm_row_slice_read(cm, start, end):
 if start==end: return _new_cm((0, cm.shape[0])) # cudamat special case workaround
 if cm.shape[1]==1 and start==0 and end==1: return cm # cudamat special case workaround
 ret = cm.get_col_slice(start, end)
 return ret

def _read_single_index(index, axisLen):
 index = int(index)
 if index>=axisLen or index<-axisLen: raise IndexError('index out of bounds. index %d requested on an axis of length %d' % (index, axisLen))
 return index % axisLen

def _short_slice(i): return slice(i, i+1)

def _read_simple_slice(sl, axisLen):
 assert sl.step in (None, 1), 'simple slice not understood'
 sFrom, sTo = slice(( None if sl.start==None else int(sl.start)), ( None if sl.stop==None else int(sl.stop))).indices(axisLen)[:2]
 if sFrom>sTo: sTo = sFrom
 return (sFrom, sTo, sTo-sFrom)

def _extend_shape(shape, nAxes): return (1,) * (nAxes-len(shape)) + shape
 



# ------------------------------------------------------------------------------- memory management

max_memory_usage = numpy.inf # public

_cmsForReuse = _collections.defaultdict(list) # dict from size to list of reusable (abandoned) cms
__memoryInUse = 0
_memoryUsers = _collections.defaultdict(lambda: (0, 0))
track_memory_usage = False

def _new_cm(sizeOrShape):
 """
 Internal.
 Returns a new CUDAMatrix object of the given size.
 This is the only proc that allocs gpu mem.
 """
 global __memoryInUse
 if type(sizeOrShape) == tuple:
  if _prodT(sizeOrShape)==0: return _new_cm(1) # cudamat workaround: cudamat can't handle size 0 arrays
  else: return _new_cm(sizeOrShape[0]*sizeOrShape[1]).reshape((sizeOrShape[1], sizeOrShape[0]))
 size = sizeOrShape
 if size==0: return _cudamat.empty((1, 1)) # cudamat workaround
 if len(_cmsForReuse[size])!=0:
  return _cm_reshape(_cmsForReuse[size].pop(), (1, size)) # re-use an abandoned cm
 _init_gpu()
 if __memoryInUse+size*4*5 > max_memory_usage: free_reuse_cache(False) # if we're somewhat close to the limit, then free what's easy to free, and hope that there are contiguous blocks available.
 if __memoryInUse+size*4 > max_memory_usage: # if we're (still) OVER the limit, then do whatever can be done to make more mem available
  free_reuse_cache(True) # gc.collect can take quite some time
  if __memoryInUse+size*4 > max_memory_usage:
   raise MemoryError('Gnumpy ran out of memory. Currently in use are %s; the maximum allowed is %s; so the present request for %s fails. Free some memory and try again.' % (_n_bytes_str(__memoryInUse), _n_bytes_str(max_memory_usage), _n_bytes_str(size*4)))
 try:
  ret = _cudamat.empty((size, 1))
  __memoryInUse += size*4 # do this only if the malloc succeeded
  return ret
 except _cudamat.CUDAMatException, e: # this means that malloc failed
  raise MemoryError('The GPU failed to allocate the requested %d bytes of memory. This doesn\'t mean that your program is using too much memory. It does, however, mean that you should reduce the value of gnumpy.max_memory_usage (currently %s), to always have some memory unused (which is necessary to find contiguous large blocks of memory to allocate). Failing to allocate enough memory makes the GPU feel very unwell, so you are advised to restart Python now, or expect to see incoherent error messages and risk causing more serious damage.' % (size*4, str(max_memory_usage)))

def free_reuse_cache(completely=True):
 """
 This frees all GPU memory that is not in use but is kept allocated for re-use.
 If <completely> is set to False, this works quicker but less thoroughly.
 """
 if completely: _gc.collect() # this has to happen before the loop, because this may add more entries in _cmsForReuse which then have to be freed by the loop
 global __memoryInUse
 for size in _cmsForReuse:
  while _cmsForReuse[size]:
   _cmsForReuse[size].pop()
   __memoryInUse -= size*4
 del _gc.garbage[:] # this shouldn't be necessary at all, but for some reason perfectly referenced AND perfectly deletable cms get put there

def _n_bytes_str(n):
 def _base(s): return ( _base(s[:-3]) + ',' + s[-3:] if len(s)>=4 else s)
 return _base(str(n)) + ' bytes'
 
def memory_in_use(in_megabytes=False):
 """ returns the number of bytes (or megabytes if you asked for that) of GPU memory that are in use. """
 return __memoryInUse // ( 2**20 if in_megabytes else 1)
   
def memory_available(free_reuse_cache_first):
 if free_reuse_cache_first: free_reuse_cache()
 return max_memory_usage - memory_in_use()

def _calling_line():
 """ Internal. Inspects the current python call stack and returns a nice string description of the line of code that called gnumpy. """
 stack = _pdb.traceback.extract_stack()[::-1] # newest first
 stack = stack[( i for i, x in enumerate(stack) if x[0] != stack[0][0]).next():] # skip any gnumpy procs on the stack
 def stackFrameToString(frame): return 'File "%s", line %d, in function %s:    %s' % (frame[0], frame[1], frame[2], ( '<command unknown>' if frame[3]==None else frame[3]))
 ret = stackFrameToString(stack[0])
 for frame in stack[1:]:
  if 'File "<ipython console>",' in ret: break
  if 'File "<stdin>",' in ret: break
  ret += '\n  Called by: ' + stackFrameToString(frame)
 return ret

def memory_allocators(minimum_n_bytes=1):
 """ Prints a list of lines in your code that caused allocated GPU memory that's still in use. """
 if not track_memory_usage:
  print 'The variable gnumpy.track_memory_usage must be set to True, to enable memory data collection (which can slow down your program a lot).'
  return
 for line, (n,amt) in sorted(_memoryUsers.items(), key=lambda x:x[1][1]) [::-1] :
  if amt >= minimum_n_bytes:
   print '%d objects, totalling %s, that are still in use, were allocated by: %s' % (n, _n_bytes_str(amt), line)
   print
 


# ------------------------------------------------------------------------------- external procs

def status():
 if _useGpu=='no': print 'gnumpy is running on the CPU, i.e. in simulation mode. The data type is float%s.' % _precision
 if _useGpu=='yes':
  if _boardId==None: print 'gnumpy is planning to run on a GPU, but hasn\'t yet chosen & initialized a board.'
  else: print 'gnumpy is running on GPU board #%d.' % _boardId
 print '%s of gpu memory are in use, of which at least %s can be freed immediately by gnumpy.free_reuse_cache().' % (_n_bytes_str(__memoryInUse), _n_bytes_str(__builtin__.sum( size*len(cms)*4 for size, cms in _cmsForReuse.items())))
 
 
  
def _rand__base(shapeInfo, distribution, zero_d_means_scalar):
 if len(shapeInfo)==1 and _isSequence(shapeInfo[0]): zero_d_means_scalar = False; shapeInfo = shapeInfo[0]
 ret = empty(shapeInfo)
 {'uniform': _cmType.fill_with_rand, 'normal': _cmType.fill_with_randn}[distribution](ret._base)
 if ret.size!=0 and _doExpensiveCheck(): assert ret.sum() < 100 + 2*ret.size, 'numerical gpu error: rand() gave a result>100'
 if len(shapeInfo) == 0 and zero_d_means_scalar: return ret.item()
 else: return ret

def tile(a, reps):
 if type(reps) in _numberTypes: reps = (reps,)
 reps = tuple(reps) # for generator expressions
 a = as_garray(a)
 if len(reps) > a.ndim: a = a._add_axes(len(reps))
 if len(reps) < a.ndim: reps = _extend_shape(reps, a.ndim) # now len(reps)==a.ndim
 retShape = tuple([ a.shape[i] * reps[i] for i in tuple(xrange(len(reps)))])
 if _prodT(retShape)==0: return zeros(retShape)
 if _prodT(reps)==1: return a
 for i in range(a.ndim-1): # merge replication requests on adjacent axes, for efficiency.
  if reps[i]!=1 and reps[i+1]!=1 and a.shape[i]==1: return a.reshape(_deleteT2(a.shape, i)).tile(reps[:i]+(_prodT(reps[i:i+2]),)+reps[i+2:]).reshape(map(op.mul, a.shape, reps))
 def dataIDone(nextA, i): return nextA.reshape(_modifyT(a.shape, i, a.shape[i]*reps[i])).tile(_modifyT(reps, i, 1))
 if reps[0]!=1: # replicating rows is easy and efficient: just repeat the data a number of times.
  temp = empty((reps[0], a.size)) # shape doesn't matter because dataIDone changes it
  tempCm = temp._base_shaped(1)
  if reps[0]>=1:
   _cm_row_slice_read(tempCm, 0, 1).assign(a._base_as_row())
   nCopiesDone = 1
   while nCopiesDone < reps[0]:
    nNow = __builtin__.min(nCopiesDone, reps[0]-nCopiesDone)
    _cm_row_slice_read(tempCm, nCopiesDone, nCopiesDone + nNow).assign(_cm_row_slice_read(tempCm, 0, nNow))
    nCopiesDone += nNow
  return dataIDone(temp, 0)
 # the general case is repeating a subset (aot the whole array) n times, before moving on to the next subset
 # using a transpose with the right shape, the subsets can become columns. those can be lengthened because that is replicating rows; a second transpose makes them now-lengthened subsets again
 axis = __builtin__.min( i for i in range(a.ndim) if reps[i]!=1)
 return dataIDone(a.reshape_2d(axis).T.tile((reps[axis], 1)).T, axis)
 
def is_garray(x): return isinstance(x, garray)
def is_array(x): return isinstance(x, garray) or type(x) == numpy.ndarray

def rand(*shapeInfo):
 """ the desired array shape can be entered either as integers or as a tuple of integers. If you enter a tuple you always get an array; if you enter nothing you get a scalar. """
 return _rand__base(shapeInfo, 'uniform', True)

def randn(*shapeInfo):
 """ the desired array shape can be entered either as integers or as a tuple of integers. If you enter a tuple you always get an array; if you enter nothing you get a scalar. """
 return _rand__base(shapeInfo, 'normal', True)

def empty(shape):
 if _isSequence(shape) or type(shape) == _types.GeneratorType: shape = tuple(shape)
 else: shape = (shape,)
 return garray(_new_cm(_prodT(shape)), shape, None)

def zeros (shape):
 if _isSequence(shape) or type(shape) == _types.GeneratorType: shape = tuple(shape)
 else: shape = (shape,)
 ret = empty(shape)
 ret._base.assign(0)
 return ret

def ones (shape):
 if _isSequence(shape) or type(shape) == _types.GeneratorType: shape = tuple(shape)
 else: shape = (shape,)
 ret = empty(shape)
 ret._base.assign(1)
 return ret

def seed_rand(seed=None):
 _init_gpu()
 if seed==None: seed = int(_time.time())
 _cudamat.CUDAMatrix.init_random(seed)

def dot(a1, a2):
 # internally: for matrix-matrix multiplies only; vectors are treated like special cases.
 a1 = as_garray(a1); a2 = as_garray(a2)
 if a1.ndim==0 or a2.ndim==0: return a1*a2
 if a1.ndim==a2.ndim==1: return dot(a1.reshape(1, a1.size), a2.reshape(a2.size, 1)).item()
 if a1.ndim==2 and a2.ndim==1: return dot(a1, a2.reshape(a2.size, 1)).ravel() # treat a2 like a column vector
 if a1.ndim==1 and a2.ndim==2: return dot(a1._add_axes(2), a2)[0]   # treat a1 like a row vector
 if a1.shape[-1] != a2.shape[-2]: raise ValueError('arrays not aligned for dot product. a dot product was requested of arrays with shapes %s and %s' % (a1.shape, a2.shape))
 if a1.ndim==a2.ndim==2:
  retShape = (a1.shape[0], a2.shape[1])
  if a1.shape[1]==0: return zeros(retShape) # cudamat bug workaround
  ret = empty(retShape)
  if ret.size!=0: _cudamat.dot(a2._base_as_2d(), a1._base_as_2d(), ret._base_as_2d())
  return ret
 if a1.ndim >= 2 and a2.ndim >= 2:
  # this is not necessarily fast, because if a2.ndim>=3 then it involves a transpose that is done by a loop
  a12 = ( a1.reshape_2d(-1) if a1.ndim!=2 else a1)
  a22 = ( a2.transpose((a2.ndim-2,) + tuple(xrange(a2.ndim-2)) + (a2.ndim-1,)).reshape_2d(1)
          if a2.ndim!=2 else
          a2)
  retShape = _deleteT2(a1.shape, -1) + _deleteT2(a2.shape, -2)
  return dot(a12, a22).reshape(retShape)
 raise NotImplementedError('dot with arguments of shapes %s and %s' % (a1.shape, a2.shape))

def outer(vec1, vec2): return dot(vec1.ravel()[:, newaxis], vec2.ravel()[newaxis, :])

def concatenate(arrays, axis=0):
 arrays = tuple(map(as_garray, arrays))
 if axis<0: axis += arrays[0].ndim
 if not _isSequence(arrays) or not type(axis) in _numberTypes: raise ValueError('wrong argument types to gnumpy.concatenate: expected <arrays> to be a sequence and <axis> to be a number, but got types %s and %s.' % (type(arrays), type(axis)))
 if axis not in range(arrays[0].ndim): raise ValueError('bad axis number (%d) specified (the first array has %d axes)' % (axis, arrays[0].ndim))
 if not _allTheSame( _deleteT2(a.shape, axis) for a in arrays): raise ValueError('array dimensions must agree except possibly for axis #%d. The given array shapes are: %s' % (axis, tuple( a.shape for a in arrays)))
 finalShape = _modifyT(arrays[0].shape, axis, __builtin__.sum( a.shape[axis] for a in arrays))
 if axis==0:
  ret = empty(finalShape)
  nextI = 0
  for a in arrays:
   _cm_row_slice_read(ret._base_shaped(ret.ndim), nextI, nextI+a.size).assign(a._base_shaped(a.ndim))
   nextI += a.size
  return ret
 else:
  return concatenate(tuple([ a.reshape_2d(axis).T for a in arrays]), 0).T.reshape(finalShape)
 
def where(a, *vararg):
 """
 Note: if only one argument is provided, the returned value will be a tuple of *numpy* arrays of integer indices (gpu arrays can only contain floats).
 """
 if vararg==_t0: return numpy.where(as_numpy_array(a))
 assert len(vararg)==2, 'wrong number of arguments to gnumpy.where()'
 return garray(numpy.where(as_numpy_array(a), as_numpy_array(vararg[0]), as_numpy_array(vararg[1])))

def nonzero(a):
 """ See notes for where(). """
 return where(a)
 
newaxis = None

def eye(n): return diagflat(ones(n))

def diagflat(a, k=0):
 if isinstance(a, garray): return a.diagflat(k)
 else: return numpy.diagflat(a,k)

def tensordot(a, b, axes=2):
 if type(axes) in _numberTypes: return dot(a.reshape_2d(a.ndim-axes), b.reshape_2d(axes)).reshape(a.shape[:a.ndim-axes] + b.shape[axes:])
 assert len(axes)==2 and len(axes[0])==len(axes[1]), 'the axes parameter to gnumpy.tensordot looks bad'
 aRemove, bRemove = (tuple(axes[0]), tuple(axes[1]))
 return tensordot(a.transpose(filter(lambda x: x not in aRemove, tuple(xrange(a.ndim))) + aRemove),
                  b.transpose(bRemove + filter(lambda x: x not in bRemove, tuple(xrange(b.ndim)))),
                  len(aRemove))

 
 
# ------------------------------------------------------------------------------- reductors

def _reductor__base(x, axis, gpuOp, npOp):
 if type(x) == numpy.ndarray: return npOp(x, axis)
 if not isinstance(x, garray): x = garray(x)
 if gpuOp==None: return garray(npOp(x.as_numpy_array(), axis))
 else: return gpuOp(x, axis)

def all(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return _reductor__base(x, axis, garray.all, numpy.all)

def any(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return _reductor__base(x, axis, garray.any, numpy.any)

def sum(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return _reductor__base(x, axis, garray.sum, numpy.sum)

def mean(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return _reductor__base(x, axis, garray.mean, numpy.mean)

def max(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return _reductor__base(x, axis, garray.max, numpy.max)

def min(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return _reductor__base(x, axis, garray.min, numpy.min)

def prod(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return _reductor__base(x, axis, None, numpy.prod)

def std(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return _reductor__base(x, axis, None, numpy.std)



# ------------------------------------------------------------------------------- elementwise operations

def _elementwise__base(x, opGpu, opNp):
 if isinstance(x, garray):
  if opGpu==None: return _check_number_types(garray(opNp(x.as_numpy_array())))
  else: return _check_number_types(opGpu(x))
 if type(x) in _numberTypes: return _check_number_types(float(opNp(x)))
 if type(x) == numpy.ndarray:
  if x.ndim==0: return _check_number_types(numpy.array(opNp(x)))
  else: return _check_number_types(opNp(x))
 raise TypeError('value %s of unexpected type %s provided to %s()' % (x, type(x), str(opNp).split("'")[1]))

def abs(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return _elementwise__base(x, garray.abs, numpy.abs)

def exp(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return _elementwise__base(x, garray.exp, numpy.exp)

def isinf(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return _elementwise__base(x, garray.isinf, numpy.isinf)

def isnan(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return _elementwise__base(x, garray.isnan, numpy.isnan)

def log(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return _elementwise__base(x, garray.log, numpy.log)

def log_1_plus_exp(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return _elementwise__base(x, garray.log_1_plus_exp, lambda x: 1.+exp(x))
 
def log10(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return _elementwise__base(x, None, numpy.log10)
 
def logistic(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return _elementwise__base(x, garray.logistic, lambda x: 1./(1. + exp(-x)))
 
def negative(x):
 """
 Like -x, except that a zero dimensional numpy array input results in a numpy array return value.
 This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats).
 """
 return _elementwise__base(x, op.neg, op.neg)

def sign(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return _elementwise__base(x, garray.sign, numpy.sign)

def sqrt(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return _elementwise__base(x, garray.sqrt, numpy.sqrt)

def tanh(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return _elementwise__base(x, garray.tanh, numpy.tanh)
 

 
 

class garray(object):
 """
 A class designed to interface like numpy arrays, and internally do its work on a GPU.
 Documentation can be found at http://www.cs.toronto.edu/~tijmen/gnumpy.html
 """

 # ------------------------------------------------------------------------------- internal aux

 def _set_shape_info(self, shape): # setting these as attributes rather than properties saves exec time
  self.shape = shape
  self.size = _prodT(shape)
  self.ndim = len(shape)
 
 @property
 def nbytes(self): return self.size * 4
 @property
 def nMBytes(self): return self.nbytes / 2**20
  
 def _base_shaped(self, nDimsAsRows): return _cm_reshape(self._base, (_prodT(self.shape[:nDimsAsRows]), _prodT(self.shape[nDimsAsRows:])))
 def _base_as_row(self): return _cm_reshape(self._base, (1, self.size))
 def _base_as_2d(self): return self._base.reshape((self.shape[1], self.shape[0])) # optimized from self._base_shaped(1) by inlining
 
 def _new_cm(self, nDimsAsRows=0): return _new_cm((_prodT(self.shape[:nDimsAsRows]), _prodT(self.shape[nDimsAsRows:]))) # same size as self, with given shape
 
 def _new(self, cm): return garray(cm, self.shape, None) # short notation for the result of elementwise ops
 
 def _tile_to_broadcast(self, otherShape, indicesToBroadcast='all'):
  """ self.shape and otherShape must already be of the same length. otherShape is relevant only where self.shape is 1. """
  if otherShape == self.shape: return self
  assert self.ndim == len(otherShape), 'dimensionality mismatch in _tile_to_broadcast'
  if indicesToBroadcast=='all': indicesToBroadcast = tuple( i for i in range(self.ndim) if self.shape[i]==1 and otherShape[i]!=1)
  return self.tile( ( 1 if i not in indicesToBroadcast else otherShape[i] ) for i in range(self.ndim))
 
 def _broadcastable_op(self, other, operator):
  """
  accepted ops: "add", "multiply", "less than", "greater than".
  other must be either scalar or garray.
  """
  basicHandler = {'add': _cmType.add, 'multiply': _cmType.mult, 'less than': _cmType.less_than, 'greater than': _cmType.greater_than, 'pow': _cudamat.pow}[operator]
  if (type(other) in _numberTypes or (other.size==1 and other.ndim <= self.ndim)): # having other be a scalar is faster than doing a broadcast
   return self._new(basicHandler(self._base_as_row(), ( other.item() if isinstance(other, garray) else other), self._new_cm()))
  if operator=='pow': raise NotImplementedError('a**b where b is anything other than a scalar')
  other = as_garray(other)
  if self.ndim > other.ndim: other = other._add_axes(self.ndim)
  if self.ndim < other.ndim: return self._add_axes(other.ndim)._broadcastable_op(other, operator)
  if operator in ('less than', 'greater than'):
   self2 = self._tile_to_broadcast(other.shape)
   return self2._new(basicHandler(self2._base_as_row(), other._tile_to_broadcast(self2.shape)._base_as_row(), self2._new_cm()))
  if self.ndim < other.ndim: return other._broadcastable_op(self, operator) # now self.ndim == other.ndim
  selfToBroadcast =  tuple( self.shape[i]==1 and other.shape[i]!=1 for i in range(self.ndim))
  otherToBroadcast = tuple( other.shape[i]==1 and self.shape[i]!=1 for i in range(self.ndim))
  bc = otherToBroadcast; bci = tuple( i for i in tuple(xrange(len(bc))) if bc[i])
  if reduce(op.or_, selfToBroadcast, False) and reduce(op.or_, otherToBroadcast, False): return self._broadcastable_op(other._tile_to_broadcast(self.shape, bci), operator)
  if reduce(op.or_, selfToBroadcast, False): return other._broadcastable_op(self, operator) # now only other may have dims that need to be broadcast
  if reduce(op.or_, ( other.shape[i] not in (1, self.shape[i]) for i in range(self.ndim)), False): raise ValueError('shape mismatch: objects cannot be broadcast to a single shape')
  if not reduce(op.or_, otherToBroadcast, False): return self._new(( _cmType.add if operator=='add' else _cmType.mult)(self._base_as_row(), other._base_as_row(), self._new_cm())) # handle case: nothing to bc
  if self.size==0: return self
  if bci == tuple(xrange(len(bci))): # handle case: only the first dims need broadcasting
   return self._new(( _cmType.add_col_vec if operator=='add' else _cmType.mult_by_col)(self._base_shaped(len(bci)), other._base_as_row(), self._new_cm(len(bci))))
  if bci == tuple(xrange(self.ndim-len(bci), self.ndim)): # handle case: only the last dims need broadcasting
   return self._new(( _cmType.add_row_vec if operator=='add' else _cmType.mult_by_row)(self._base_shaped(self.ndim-len(bci)), other._base_shaped(self.ndim-len(bci)), self._new_cm(self.ndim-len(bci))))
  # remaining case: broadcasting neither just the first dims nor just the last dims. this can be done very intelligently, but for now I won't bother
  return self._broadcastable_op(other._tile_to_broadcast(self.shape, bci[:1]), operator)

 def _elementwise_unary(self, handler): return _check_number_types(self._new(handler(self._base_as_row(), self._new_cm())))

 def _reduction__base(self, operator, axis):
  if axis==None and operator==_cmType.sum and self.size==0: return 0.0 # cudamat bug workaround
  if axis==None: return self.ravel()._reduction__base(operator, 0).item()
  if not type(axis) in _numberTypes: raise TypeError('the value %s is not appropriate for the "axis" parameter.' % str(axis))
  axis = int(axis)
  if axis < -self.ndim or axis>=self.ndim: raise ValueError('axis (%d) out of bounds for an array with %d axes.' % (axis, self.ndim))
  axis %= self.ndim
  if axis==0 and operator==_cmType.max: # max over rows is not yet supported in cudamat
   return self.reshape_2d(1).T.max(1).reshape(self.shape[1:])
  if axis==0 and self.ndim==1 and self.size>5000 and operator==_cmType.sum: # optimization. apparently, cudamat is not maximally efficient.
   n = int(numpy.sqrt(self.size-1))
   return self[:n*n].reshape((n, n))._reduction__base(operator, 0)._reduction__base(operator, 0) + self[n*n:]._reduction__base(operator, 0)
  if operator==_cmType.sum:
   chunkSize = 1024*256 # sum over longer dimensions fails in cudamat
   nChunks = (self.shape[axis] + chunkSize-1) // chunkSize
   if nChunks>1:
    return reduceAdd( self[(slice(None),) * axis + (slice(chunkI*chunkSize, __builtin__.min(self.shape[axis], (chunkI+1)*chunkSize)),)]._reduction__base(operator, axis)
                      for chunkI in range(nChunks))
  if self.shape[axis]==0:
   if operator==_cmType.max: raise ValueError('max over an axis of length 0 is undefined')
   assert operator==_cmType.sum, 'unexpected operator'
   return zeros(_deleteT2(self.shape, axis))
  if self.size==0 and self.shape[axis]!=0: return empty(_deleteT2(self.shape, axis))
  if axis==0: return _check_number_types(garray(operator(self._base_shaped(1), 1, _new_cm(_prodT(self.shape[1:]))), self.shape[1:], None))
  if axis==self.ndim-1:
   if self.ndim!=2: return self.reshape_2d(-1)._reduction__base(operator, 1).reshape(self.shape[:-1])
   if self.ndim==2:
    chunkSize = 2**16-1
    nChunks = (len(self) + chunkSize-1) // chunkSize
    if nChunks>1: # cudamat chokes on big arrays, so break it in pieces for cudamat
     chunks = tuple( self[chunkI*chunkSize : __builtin__.min((chunkI+1)*chunkSize, len(self))]
                     for chunkI in range(nChunks))
     return concatenate( chunk._reduction__base(operator, 1) for chunk in chunks)
    else: # small array
     return _check_number_types(garray(operator(self._base_shaped(1), 0, _new_cm((len(self), 1))), (len(self),), None))
  return self.transpose_simple(axis)._reduction__base(operator, 0).transpose_simple(-axis)
 

 
 # ------------------------------------------------------------------------------- external misc non-numerical
 
 def __init__(self, data, copy=True, ndmin=0):
  """ the parameters mean the same as in numpy.array() """
  if type(data)!=_cmType: assert copy in (True, False) and type(ndmin) in _numberTypes, 'garray() parameters copy=%s, ndmin=%s are not of the right type' % (str(copy), str(ndmin))
  if type(data)==_cmType: # internal use only. the 3 arguments are, unlike their names suggest, the ._base, .shape, ._is_alias_of
   self._base = data
   self._set_shape_info(copy)
   self._is_alias_of = ndmin
   if self._is_alias_of==None and track_memory_usage:
    self.allocating_line = _calling_line()
    _memoryUsers[self.allocating_line] = (_memoryUsers[self.allocating_line][0]+1, _memoryUsers[self.allocating_line][1]+self.size*4)
  elif isinstance(data, garray):
   if ndmin>0: data = data._add_axes(ndmin)
   garray.__init__(self, 
    ( _new_cm(data.size).assign(data._base_as_row()) if copy else data._base),
    data.shape,
    ( None if copy else data))
  elif type(data) == _types.GeneratorType: garray.__init__(self, tuple(data), ndmin=ndmin)
  elif _isSequence(data):
   if len(data)==0 or not _any2_(data, is_garray): garray.__init__(self, numpy.array(data, ndmin=ndmin), copy=False)
   else: garray.__init__(self, concatenate( as_garray(element)[None] for element in data), ndmin=ndmin) # no need to copy, because concat copies.
  else: # remaining cases. essentially init from numpy array.
   npa = numpy.array(data, copy=False) # in case data was a number
   if str(npa.dtype) in ('object', '|S3'): raise TypeError('Cannot convert "%s" to a garray.' % data) 
   # we're not using the cudamat constructor, because that always allocs gpu mem, and this way the mem may come from re-use.
   cm = _new_cm(npa.size)
   if not hasattr(cm, 'numpy_array'): cm.copy_to_host() # if cm was created using cudamat.empty, this is needed to associate cm with a numpy array
   if npa.size!=0: cm.numpy_array[:] = npa.reshape((-1, 1), order='C') # no cudamat.reformat is needed, because that's only dtype and order change, which are handled by the assignment anyway
   cm.copy_to_device()
   garray.__init__(self, cm, _extend_shape(npa.shape, ndmin), None)

 def __new__(cls, *args, **kwarg): return object.__new__(cls)
   
 def as_numpy_array(self, dtype=numpy.float64):
  if self.size==0: return numpy.zeros(self.shape, dtype)
  return numpy.array(self._base_as_row().asarray(), copy=True, order='C', dtype=dtype).reshape(self.shape)
 
 asarray = as_numpy_array # the cudamat name
 
 def astype(self, type): return self.asarray().astype(type)
 
 tile = tile
 
 def ravel(self): return self.reshape(-1)
 
 def item(self): return self.as_numpy_array().item()
 
 def _add_axes(self, finalNdim): return self.reshape(_extend_shape(self.shape, finalNdim))

 def sort(self, axis=-1, kind='quicksort', order=None):
  """ like numpy.sort, this sorts in place and returns None. """
  temp = self.as_numpy_array()
  temp.sort(axis, kind, order)
  self[:] = temp
 
 def reshape(self, *newShape):
  if len(newShape)==1 and not type(newShape[0]) in _numberTypes: newShape = tuple(newShape[0])
  if not _all2_(newShape, _isNumber): raise TypeError('the parameters to reshape don\'t look like a valid shape')
  if -1 in newShape:
   if _prodT(newShape)==0: raise ValueError("-1 as a parameter to reshape is not allowed if one of the other parameters is zero.")
   newShape = _modifyT(newShape, op.indexOf(newShape, -1), self.size//-_prodT(newShape))
  if _prodT(newShape) != self.size: raise ValueError('the total number of items cannot be changed in a reshape')
  return garray(self._base, newShape, self)
 
 def reshape_2d(self, n_dimensions_as_rows):
  """ reshapes to 2 axes. The first <n_dimensions_as_rows> axes of the array become the first axis of the returned value. The remaining ones form the second axis. """
  if n_dimensions_as_rows<0: n_dimensions_as_rows += self.ndim
  return self.reshape((_prodT(self.shape[:n_dimensions_as_rows]), _prodT(self.shape[n_dimensions_as_rows:])))
 
 @property
 def T(self):
  if self.ndim==2: # _base case
   if self.size==0: return self.reshape(tuple(reversed(self.shape))) # cudamat bug workaround
   if self.shape[1]>1e6: # cudamat bug workaround. with 2m columns it fails
    return concatenate( self[:, i*10**6 : (i+1)*10**6].T for i in range((self.shape[1]+10**6-1)//10**6))
   if self.shape[0]>1e6: # cudamat bug workaround. using concat is not an option, because that uses transpose.
    ret = empty(tuple(reversed(self.shape)))
    for i in range((self.shape[0]+10**6-1)//10**6):
     ret[:, i*10**6 : (i+1)*10**6] = self[i*10**6 : (i+1)*10**6].T 
    return ret
   return garray(self._base_as_2d().transpose(_new_cm(tuple(reversed(self.shape)))), tuple(reversed(self.shape)), None)
  else: return self.transpose()  

 def transpose_simple(self, nDimsToGroup):
  """ shifts the first <nDimsToGroup> axes to the end, and the remaining ones to the start. This returns a new array, not an alias. """
  if nDimsToGroup<0: nDimsToGroup += self.ndim
  return self.reshape_2d(nDimsToGroup).T.reshape(self.shape[nDimsToGroup:] + self.shape[:nDimsToGroup])
 
 def transpose(self, *axes):
  """ like numpy.transpose, except that this doesn't return an alias, but rather a new array. """
  if len(axes)==1 and not type(axes[0]) in _numberTypes: axes = tuple(axes[0])
  if axes==_t0: axes = tuple(reversed(tuple(xrange(self.ndim))))
  if axes == tuple(xrange(self.ndim)): return self.copy()
  if tuple(sorted(axes)) != tuple(xrange(self.ndim)): raise ValueError("%s is not a valid argument to transpose() of an array of %d axes" % (axes, self.ndim))
  for i in range(self.ndim-1): 
   if axes[i+1]==axes[i]+1: return (self. # see if the task can be simplified by collapsing some axes that are kept adjacent
    reshape(self.shape[:axes[i]] + (_prodT(self.shape[axes[i]:axes[i]+2]),) + self.shape[axes[i]+2:]).
    transpose((originalAxisI-(originalAxisI>axes[i])) for originalAxisI in _deleteT2(axes, i+1)).
    reshape(self.shape[axisI] for axisI in axes))
  def shiftAxesRight(shiftN): return self.transpose_simple(-shiftN).transpose( (axisI+shiftN)%self.ndim for axisI in axes)
  for i in range(self.ndim-1): # see if the task can be simplified by rotating axes right by 1. if so, the loop before this one can simplify further
   if axes[i:i+2] == (self.ndim-1, 0): return shiftAxesRight(1)
  # no further simplifications can be done. we need to proceed with a loop over the first axis. First rotate the intended axis to position 0.
  if axes[0]!=0: return shiftAxesRight(-axes[0])
  ret = empty( self.shape[axisI] for axisI in axes)
  for i in range(self.shape[0]): ret[i] = self[i].transpose( x-1 for x in axes[1:])
  return ret
   
 def copy(self): return garray(self, copy=True)
 
 def diagflat(self, k=0):
  if self.ndim!=1: return self.ravel().diagflat(k)
  if k!=0: raise NotImplementedError('k!=0 for garray.diagflat')
  ss = self.size
  ret = zeros((ss, ss))
  ret.ravel()[:-1].reshape((ss-1, ss+1))[:, 0] = self[:-1]
  if ss!=0: ret.ravel()[-1] = self[-1]
  return ret
   
 def diagonal(self):
  if self.ndim==1: return self.diagflat()
  if self.ndim==2:
   if self.shape[0] > self.shape[1]: return self[:self.shape[1]].diagonal()
   if self.shape[1] > self.shape[0]: return self[:, :self.shape[0]].diagonal()
   return self.ravel()[::self.shape[0]+1]
  raise NotImplementedError('garray.diagonal for arrays with ndim other than 1 or 2.')
 def diag(self): return self.diagonal()
  


 # ------------------------------------------------------------------------------- elementwise type checking
 
 def all_real(self):
  """ returns True iff all array elements are regular floats, as opposed to inf's, -inf's, and NaN's.  """
  return (self*0).sum()==0
  
 def isinf(self):
  """ elementwise, checking for inf or -inf. """
  return 1 - self.isreal() - self.isnan()
 
 def isreal(self):
  """ elementwise, checking for real numbers. See also .all_real() """
  return (self<numpy.inf) * (self>-numpy.inf)
 
 def isnan(self): 
  """ elementwise, checking for NaN's. """
  return (self>0) + (self<1) < .5

 def isnumber(self):
  """ elementwise, checking for anything other than NaN's """
  return (self>0) + (self<1) > .5
 
 
 
 # ------------------------------------------------------------------------------- external misc numerical
 
 def __abs__(self): return self._elementwise_unary(_cudamat.abs)
 def abs(self): return __builtin__.abs(self)
 def as_bool(self): return self!=0
 def exp(self): return self._elementwise_unary(_cudamat.exp)
 def log(self): return self._elementwise_unary(_cudamat.log)
 def log_1_plus_exp(self): return self._elementwise_unary(_cudamat.log_1_plus_exp)
 def logistic(self): return self._elementwise_unary(_cudamat.sigmoid)
 sigmoid = logistic
 def sign(self): return self._elementwise_unary(_cmType.sign)
 def sqrt(self): return self._elementwise_unary(_cudamat.sqrt)
 def tanh(self): return self._elementwise_unary(_cudamat.tanh)
 

 def sum(self, axis=None): return self._reduction__base(_cmType.sum, axis)
 def mean(self, axis=None): return self.sum(axis) / ( self.size if axis==None else self.shape[axis])
 def max(self, axis=None):
  if self.isnan().any2():
   if axis==None: return self.asarray().max()
   else: return garray(self.asarray().max(axis))
   #raise NotImplementedError('cudamat max fails with nans')
  return self._reduction__base(_cmType.max, axis)
 def argmax(self, axis=None): return numpy.argmax(self.asarray(), axis)
 def argmin(self, axis=None): return numpy.argmin(self.asarray(), axis)
 def min(self, axis=None): return -(-self).max(axis)
 def all(self, axis=None): return ( True if self.size==0 else (self.as_bool()).min())
 def any(self, axis=None): return ( False if self.size==0 else (self.as_bool()).max())
 
 def all2(self, axis=None): return 1-(1-self).any2(axis)  # optimized for when I'm sure that the content is boolean
 def any2(self, axis=None): return self.sum(axis) > 0  # optimized for when I'm sure that the content is boolean
 
 def rand(self, distribution = 'uniform'):
  """
  returns a new garray, of the same shape as self, filled with random numbers.
  <distribution> can be either 'uniform' or 'normal'.
  """
  return _rand__base(self.shape, distribution, False)

 def euclid_norm(self): return self._base.euclid_norm()

 dot = dot
 where = where
 nonzero = nonzero
 
 def __nonzero__(self): return self.size==1 and self.item()!=0
 
 
 # ------------------------------------------------------------------------------- operator overloads, numerical
 
 def __add__(self, other): return _check_number_types(self._broadcastable_op(as_garray_or_scalar(other), 'add'))
 def __mul__(self, other): return _check_number_types(self._broadcastable_op(as_garray_or_scalar(other), 'multiply'))
 def __or__(self, other): return (self.as_bool() + other.as_bool()).as_bool()
 def __and__(self, other): return self.as_bool() * other.as_bool()
 
 def __pow__(self, other, modulo=None):
  if modulo!=None: raise NotImplementedError('power with modulo')
  if type(other) in _numberTypes and other==2: return self*self # faster
  return self._broadcastable_op(as_garray_or_scalar(other), 'pow')
 
 
 # the following would be a lot simpler if I wouldn't have to deal with nans
 
 def __lt__(self, other): return _check_number_types(self._broadcastable_op(as_garray_or_scalar(other), 'less than'))
 
 def __gt__(self, other): return _check_number_types(self._broadcastable_op(as_garray_or_scalar(other), 'greater than'))
 
 def __le__(self, other): return self.isnumber() * as_garray(other).isnumber() * (1-(self>other))
 
 def __ge__(self, other): return self.isnumber() * as_garray(other).isnumber() * (1-(self<other))
 
 def __ne__(self, other): return ( 1-(self==other) if type(other) in _castableTypes else True)
 
 def __eq__(self, other): return ( (self<=other) * (self>=other) if type(other) in _castableTypes else False)
 
 def eq2(self, other):
  """
  Returns a boolean: True if self and other are the same (arrays with the same shape and contents); False otherwise.
  This is what == does on most Python objects (on arrays it's been strangely overloaded though).
  garrays compare equal to numpy arrays with the same contents, even if the data types differ.
  """
  if self is other: return True
  if not is_array(other): return False
  if self.shape != other.shape: return False
  return all(self==other)==1
 
 def __sub__(self, other):
  if isinstance(other, garray) and other.shape==self.shape: # use specialized method
   return self._new(self._base_as_row().subtract(other._base_as_row(), self._new_cm()))
  else: return self + -as_garray(other) # if i need to broadcast, making use of the row add and col add methods is probably faster
 
 def __div__(self, other):
  if type(other) in _numberTypes: return self * (1./other)
  other = as_garray(other)
  return self * other._new(other._base_as_row().reciprocal(other._new_cm()))

 def __rmul__(self, other): return self*other
 def __radd__(self, other): return self+other
 def __rsub__(self, other): return other + -self
 def __rdiv__(self, other): return as_garray(other) / self
 def __rpow__(self, other): raise NotImplementedError('a**b where only b is a garray')
 
 def __pos__(self): return self
 def __neg__(self): return self*-1
 
 def __iadd__(self, other): self[_t0] = self+other; return self # not as direct as it might have been, but the effect is the same. "self[:]" doesn't work for 0das.
 def __imul__(self, other): self[_t0] = self*other; return self
 def __isub__(self, other): self[_t0] = self-other; return self
 def __idiv__(self, other): self[_t0] = self/other; return self
 def __imod__(self, other): self[_t0] = self%other; return self
 def __ipow__(self, other, modulo=None): self[_t0] = self.__pow__(other, modulo); return self


 
 # ------------------------------------------------------------------------------- operator overloads, non-numerical
 
 def __len__(self):
  if self.ndim==0: raise TypeError('len() of unsized object')
  return self.shape[0]
 
 def __getitem__(self, selectors):
  selectors = _nonSeqAsS(selectors)
  for i,sel in enumerate(selectors): # deal with newaxis and ellipsis
   if sel is Ellipsis: return self[selectors[:i] + (slice(None),)* (self.ndim - (__builtin__.sum( x != None for x in selectors)-1)) + selectors[i+1:]] # sel==Ellipsis is bad when sel is an array
   if sel is newaxis: return self.reshape(_insertT(self.shape, i, (1,)))[_modifyT(selectors, i, slice(None))]
  if len(selectors) > self.ndim: raise IndexError('more indices than axes')
  if _all2_(selectors, _isFullSlice): return self
  if reduce(op.and_, ( _isSequence(sel) or is_array(sel) for sel in selectors), True) and len(selectors)>=2:
   selectors = tuple(map(as_garray, selectors))
   if reduce(op.or_, ( (sel < 0).sum() > 0 for sel in selectors), False): raise NotImplementedError('negative indices in index arrays, combined with having multiple indices arrays')
   # flatten the first two dimensions into one, and translate the corresponding indices arrays into one accordingly
   return self.reshape((self.shape[0]*self.shape[1],) + self.shape[2:])[(selectors[0]*self.shape[1]+selectors[1],) + selectors[2:]]
  if __builtin__.sum( _isSequence(sel) or is_array(sel) for sel in selectors)>1:
   raise NotImplementedError('slicing with more than one sequence/array among the indices, with also other kinds of values among the indices')
  # handle the operations on different axes one by one; earlier axes are handled earlier
  axisI = ( i for i, x in enumerate(selectors) if not _isFullSlice(x)).next()
  axisLen = self.shape[axisI]
  axisSelector = selectors[axisI]
  if not _all2_(selectors[axisI+1:], _isFullSlice): return self[selectors[:axisI+1]][(slice(None),)*(axisI+(not type(axisSelector) in _numberTypes)) + selectors[axisI+1:]] # first select on axisI only; then do the further axes.
  # from here, axisI is the only axis on which we don't take a full slice
  if type(axisSelector) == _types.SliceType and axisSelector.step not in (1, None): axisSelector = numpy.arange(axisLen)[axisSelector]
  if type(axisSelector) in _numberTypes: # selecting a single location on axisI, and thus reducing the dimensionality by 1
   ret = self[selectors[:axisI] + (_short_slice(_read_single_index(axisSelector, axisLen)),)]  .reshape(_deleteT2(self.shape, axisI))
   return ( ret.item() if ret.shape==_t0 else ret) # exception, to have the same behavior as numpy
  if _isSequence(axisSelector) or type(axisSelector) == numpy.ndarray: axisSelector = garray(axisSelector)
  if isinstance(axisSelector, garray):
   # a 1d index means re-arranging this axis. I.e. a number of length 1 selections on this axis, concatenated on this axis.
   # other dimensionality means using the flattened version, and then reshaping to reflect the selector dimensionality
   if hasattr(_cmType, 'select_columns'):
    if axisI==0:
     if _doExpensiveCheck() and (axisSelector> len(self)-.5).sum() !=0: raise IndexError('index %d (found in an indices array) is too large, for an axis of length %d' % (max(axisSelector), len(self)))
     if _doExpensiveCheck() and (axisSelector<-len(self)-.5).sum() !=0: raise IndexError('index %d (found in an indices array) is too small, for an axis of length %d' % (min(axisSelector), len(self)))
     return garray(self._base_shaped(1).select_columns(axisSelector._base_shaped(axisSelector.ndim), _new_cm((axisSelector.size, self.size/self.shape[0]))), axisSelector.shape + self.shape[1:], None)
    else: return self.transpose_simple(axisI)[axisSelector].transpose_simple(-axisI)
   else: return (concatenate(tuple( self[_modifyT(selectors, axisI, slice(choiceOnThisAxis, choiceOnThisAxis+1))] for choiceOnThisAxis in axisSelector.ravel()), axisI)
                 .reshape(self.shape[:axisI] + axisSelector.shape + self.shape[axisI+1:]))
  if not type(axisSelector) == _types.SliceType: raise ValueError('index not understood: %s' % axisSelector)
  # from here, selector is a simple slice
  sFrom, sTo, sLen = _read_simple_slice(axisSelector, axisLen)
  retShape = _modifyT(self.shape, axisI, sLen)
  if _prodT(retShape)==0: return zeros(retShape)
  if axisI==0: return garray(_cm_row_slice_read(self._base_shaped(1), sFrom, sTo), retShape, self) # slice on axis 0 is free, using _cm_row_slice_read
  if axisI!=1: return self.reshape((_prodT(self.shape[:axisI]),) + self.shape[axisI:])[:, sFrom:sTo].reshape(retShape) # redirect: collapse earlier axes into one
  if self.ndim != 2: return self.reshape_2d(1)[:, sFrom * _prodT(self.shape[axisI+1:]) : sTo * _prodT(self.shape[axisI+1:])].reshape(retShape) # redirect: use long elements
  chunkSize = int(2e6)
  nChunks = (len(self) + chunkSize - 1) // chunkSize
  if nChunks>1: return concatenate( tuple(self[chunkI*chunkSize : (chunkI+1)*chunkSize, sFrom:sTo] for chunkI in range(nChunks)), 0) # redirect in batches, bc cudamat chokes on big jobs
  # _base case for column slice
  retCm = _new_cm(retShape)
  _cm_col_slice_read(self._base_shaped(1), sFrom, sTo, retCm)
  return garray(retCm, retShape, None)

 def __iter__(self):
  for i in tuple(xrange(len(self))): yield self[i]
 
 def __setitem__(self, selectors, other):
  # this is different from getitem. There, I can handle the axes one at a time. Here, it's more integrated.
  selectors = _nonSeqAsS(selectors)
  for i,sel in enumerate(selectors): # deal with ellipsis
   if sel is Ellipsis: return self.__setitem__(selectors[:i] + (slice(None),)* (self.ndim - (len(selectors)-1)) + selectors[i+1:], other) # sel==Ellipsis is bad when sel is an array
  if len(selectors) > self.ndim: raise IndexError('more indices than axes')
  if reduce(op.and_, ( is_array(sel) or _isSequence(sel) for sel in selectors), True) and selectors!=_t0:
   if len(selectors)==1:
    if not hasattr(_cmType, 'set_selected_columns'):
     raise NotImplementedError("slice assign with a sequence/array as index. Get the newest version of cudamat (or npmat if you're running on the cpu).")
    sel = as_garray(selectors[0])
    if len(sel) != len(other): raise ValueError('number of rows to set != number of provided rows')
    if other.shape[1:] != self.shape[1:]: raise ValueError('shape mismatch in assignment')
    if sel.ndim!=1: raise NotImplementedError('assignment with as index an array of ndim!=1')
    if sel.size==0: return # the current implementation of set_selected_columns doesn't handle that well
    self._base_shaped(1).set_selected_columns(sel._base_shaped(1), other._base_shaped(1))
   else: # >1 selectors, all arrays/sequences. flatten the first dimension of self, and correspondingly unify the first two selectors
    self.reshape((_prodT(self.shape[:2]),) + self.shape[2:])[(as_garray(selectors[0])*self.shape[1]+as_garray(selectors[1]),) + selectors[2:]] = as_garray(other)
   return
  if reduce(op.or_, ( _isSequence(axisSel) or is_array(axisSel) for axisSel in selectors), False): raise NotImplementedError('slice assign with a sequence/array as index, as well as other indexing objects')
  if reduce(op.or_, ( type(axisSel) == _types.SliceType and axisSel.step not in (1, None) for axisSel in selectors), False): raise NotImplementedError('slice assign with stride != 1')
  if not reduce(op.and_, ( type(axisSel) in _numberTypes or type(axisSel) == _types.SliceType for axisSel in selectors), True): raise ValueError('index not understood, in slice assignment.')
  selectors = selectors + (slice(None),)*(self.ndim-len(selectors))
  # now len(selectors) == ndim, and all selectors are single indices or simple slices
  # task: broadcast other, and do shape check.
  other = as_garray_or_scalar(other)
  assignedShape = tuple( _read_simple_slice(axisSel, self.shape[axisI])[2] for axisI, axisSel in enumerate(selectors) if not type(axisSel) in _numberTypes)
  if isinstance(other, garray):
   if other.ndim < len(assignedShape): other = other._add_axes(len(assignedShape))
   if other.ndim > len(assignedShape):
    if _prodT(other.shape[: other.ndim-len(assignedShape)]) != 1: raise ValueError('Incompatible shapes in slice assign: the assigned area has shape %s, and the incoming values have shape %s.' % (assignedShape, other.shape))
    other = other.reshape(other.shape[-len(assignedShape):])
   # now other.ndim == len(assignedShape)
   if not reduce(op.and_, ( other.shape[axisNr] in (1, assignedShape[axisNr]) for axisNr in tuple(xrange(len(assignedShape)))), True):
    raise ValueError('Incompatible shapes in slice assign: the incoming values have shape %s, but the assigned area has shape %s.' % (other.shape, assignedShape))
   other = other._tile_to_broadcast(assignedShape)
  # the only time I can use scalar assign is when I don't need cudamat's column assign at all. that only happens when all selectors other than optionally the first are full slices.
  if _all2_(selectors[1:], _isFullSlice):
   ( _cm_row_slice_read(self._base_shaped(1), _read_single_index(selectors[0], self.shape[0]), _read_single_index(selectors[0], self.shape[0])+1)
     if self.ndim==1 and type(selectors[0]) in _numberTypes else
     self[selectors[:1]]._base_as_row() # I want this to work even when selectors = _t0
     ).assign( other if type(other) in _numberTypes else other._base_as_row())
   return
  if type(other) in _numberTypes: other = garray(other)._add_axes(len(assignedShape))._tile_to_broadcast(assignedShape)  
  # now other is a garray of exactly the expected shape, and there are things other than complete slices beyond axis #0 so I'm going to need a col assign.
  # task: get rid of single indices in selectors
  for i in range(self.ndim):
   if type(selectors[i]) in _numberTypes:
    selectors = _modifyT(selectors, i, _short_slice(_read_single_index(selectors[i], self.shape[i])))
    other = other.reshape(_insertT(other.shape, i, (1,)))
  if not _isFullSlice(selectors[0]): return self[selectors[0]].__setitem__((slice(None),) + selectors[1:], other)
  # now all selectors are either full or simple slices; axis 0 is a full slice; and at least one other axis is a simple slice.
  axisI = ( i for i, x in enumerate(tuple( not _isFullSlice(sel) for sel in selectors)) if x).next()
  if _all2_(selectors[axisI+1:], _isFullSlice): # then do a column slice assign directly using cudamat.
   sFrom, sTo = _read_simple_slice(selectors[axisI], self.shape[axisI])[:2]
   elementWidth = _prodT(self.shape[axisI+1:])
   if other.size!=0: # cudamat chokes on that
    _cm_col_slice_write(self._base_shaped(axisI), sFrom*elementWidth, sTo*elementWidth, other._base_shaped(axisI))
   return
  # remaining case: there are multiple non-full slices, and the slice on axis 0 is full. strategy: transpose to bring one of those non-full slices to the front.
  selfT = self.transpose_simple(axisI)
  selfT[selectors[axisI:] + selectors[:axisI]] = other.transpose_simple(axisI)
  self._base_as_row().assign(selfT.transpose_simple(self.ndim-axisI)._base_as_row())

  

 # ------------------------------------------------------------------------------- external, but not for user to see

 def __getstate__(self):
  return (self.shape, self._base_as_row().asarray())
 
 def __setstate__(self, state):
  garray.__init__(self, state[1])
  self._set_shape_info(state[0])

 def __array__(self, *dtype):
  _envInstruction = _os.environ.get('GNUMPY_IMPLICIT_CONVERSION', 'refuse')
  assert _envInstruction in ('allow', 'warn', 'refuse'), "environment variable GNUMPY_IMPLICIT_CONVERSION, if present, should be one of 'allow', 'warn', 'refuse'."
  if _envInstruction=='refuse': raise TypeError("garray objects cannot be quietly converted to numpy arrays, because the environment variable GNUMPY_IMPLICIT_CONVERSION is set to 'refuse', or is not set at all (the default is 'refuse'). Set that variable to 'allow' or 'warn' if you wish to allow quiet conversion. garray's can always be explicitly converted using the .as_numpy_array() method.")
  if _envInstruction=='warn': print "gnumpy: warning: a garray object is being quietly converted to a numpy array, and the environment variable GNUMPY_IMPLICIT_CONVERSION is set to 'warn'. garray objects can be explicitly converted using the .as_numpy_array() method."
  return self.as_numpy_array().__array__(*dtype)
  
 def __repr__(self): return self.as_numpy_array().__repr__().replace('array(', 'garray(').replace('\n', '\n ').replace(', dtype=float32', '').replace(', dtype=float64', '') # 64 happens for empty arrays
  
 def __del__(self):
  if not hasattr(self, '_is_alias_of'):
   if _os.environ['USER']=='tijmen': print 'gnumpy cleaning up an unfinished garray. mem counting may be off now.'
   return # this object was never finished, because an exception (error or interrupt) occurred in the constructor. This check avoids error messages.
  if self._is_alias_of is None:
   # this is not true in one case: if a reference to self._base is stored somewhere explicitly (somewhere outside self but not in another garray). This happens internally sometimes. I saw it happening on the last line of setitem: a transpose is created (transposes own their mem, are not aliases), and then it's dropped but _base (obtained by _base_as_row) is still in use for a cm assign call. assert _sys.getrefcount(self._base)==2, _sys.getrefcount(self._base)
   _cmsForReuse[self.size].append(self._base)
   if track_memory_usage: _memoryUsers[self.allocating_line] = (_memoryUsers[self.allocating_line][0]-1, _memoryUsers[self.allocating_line][1]-self.size*4)
  else:
   assert type(self._is_alias_of).__name__ == 'garray', '_is_alias_of is of unexpected type, of which the str() is: "%s"' % str(type(self._is_alias_of))
   # del self._base # this is only to make the refcount assert not fail



   
_castableTypes = _numberTypes | set([tuple, list, numpy.array, garray])

