import os
use_gpu = os.environ.get('USE_GPU', 'auto')
assert use_gpu in ['auto', 'yes', 'no'], "environment variable USE_GPU, should be one of 'auto', 'yes', 'no'."
if use_gpu == 'auto':
  try:
    import cudamat as cm
    use_gpu = 'yes'
  except:
    print 'Failed to import cudamat. Using eigenmat. No GPU will be used.'
    use_gpu = 'no'
if use_gpu == 'yes':
  import cudamat as cm
  from cudamat import cudamat_conv as cc
  from cudamat import gpu_lock
elif use_gpu == 'no':
  import eigenmat as cm
