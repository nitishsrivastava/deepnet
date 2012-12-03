try:
    from cudamat_conv import *
except OSError:
    print 'cannot import cudamat_conv. Entering simulation-mode with loop. EXTREMELY SLOW!'
    from cudamat_conv_py import *
