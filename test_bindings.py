import numpy as np
from timeit import default_timer as timer
import sys
sys.path.append('bin')

import intimg

# python implementation
def pyintimg(array):
    array = np.cumsum(array, axis=0)
    array = np.cumsum(array, axis=1)
    return array

# test array
# array = np.ones([1024,1024], dtype=int)
array = np.ones([4096]*2, dtype=int)

# compute results
ref = pyintimg(array)

repetitions = 5
times = []
start = timer()
for _ in range(repetitions):
    res_cpu = intimg.intimg(array)
times.append((timer()-start)/repetitions)

start = timer()
for _ in range(repetitions):
    res_omp = intimg.intimg_parallel(array)
times.append((timer()-start)/repetitions)

start = timer()
for _ in range(repetitions):
    res_gpu1 = intimg.intimg_cuda1(array)
times.append((timer()-start)/repetitions)

start = timer()
for _ in range(repetitions):
    res_gpu2 = intimg.intimg_cuda2(array)
times.append((timer()-start)/repetitions)

# check correctness

print(f'CPU test ({times[0]*1000:.3f} ms): ', 'not ' if not np.all(ref == res_cpu) else '', 'passed.')
print(f'OMP test ({times[1]*1000:.3f} ms): ', 'not ' if not np.all(ref == res_omp) else '', 'passed.')
print(f'GPUv1 test ({times[2]*1000:.3f} ms): ', 'not ' if not np.all(ref == res_gpu1) else '', 'passed.')
print(f'GPUv2 test ({times[3]*1000:.3f} ms): ', 'not ' if not np.all(ref == res_gpu2) else '', 'passed.')
# if np.all(ref == res_omp):
#     print ('CPU-OMP test passed!')
# else:
#     print('CPU-OMP test not passed!')
#
# if np.all(ref == res_gpu1):
#     print ('GPUv1 test passed!')
# else:
#     print('GPUv1 test not passed!')
#
# if np.all(ref == res_gpu2):
#     print ('GPUv2 test passed!')
# else:
#     print('GPUv2 test not passed!')
