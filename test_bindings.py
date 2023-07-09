import numpy as np
import sys
sys.path.append('bin')

import intimg

# python implementation
def pyintimg(array):
    array = np.cumsum(array, axis=0)
    array = np.cumsum(array, axis=1)
    return array

# test array
array = np.ones([1024,1024], dtype=int)

# compute results
ref = pyintimg(array)
res_cpu = intimg.intimg(array)
res_omp = intimg.intimg_parallel(array)
res_gpu1 = intimg.intimg_cuda1(array)
res_gpu2 = intimg.intimg_cuda2(array)

# check correctness
if np.all(ref == res_cpu):
    print ('CPU test passed!')
else:
    print('CPU test not passed!')

if np.all(ref == res_omp):
    print ('CPU-OMP test passed!')
else:
    print('CPU-OMP test not passed!')

if np.all(ref == res_gpu1):
    print ('GPUv1 test passed!')
else:
    print('GPUv1 test not passed!')

if np.all(ref == res_gpu2):
    print ('GPUv2 test passed!')
else:
    print('GPUv2 test not passed!')
