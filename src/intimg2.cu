#include "intimg2.cuh"
using namespace cub;
#include <stdexcept>

// BlockScan code from:
// https://nvlabs.github.io/cub/example_block_scan_8cu-example.html
template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockScanAlgorithm ALGORITHM>
__global__
void BlockPrefixSumKernel(int* d_in, int* d_out, int w, int h)
{
    // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    // Specialize BlockStore type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
    // Specialize BlockScan type for our thread block
    typedef BlockScan<int, BLOCK_THREADS, ALGORITHM> BlockScanT;
    // Shared memory
    __shared__ union TempStorage
    {
        typename BlockLoadT::TempStorage    load;
        typename BlockStoreT::TempStorage   store;
        typename BlockScanT::TempStorage    scan;
    } temp_storage;
    // Per-thread tile data
    int data[ITEMS_PER_THREAD];

    int index = blockIdx.x;
    int stride = gridDim.x;
    for (int i = index; i < h; i += stride) {
      int* d_in_block = &d_in[i*w];
      int* d_out_block = &d_out[i*w];
      // Load items into a blocked arrangement
      BlockLoadT(temp_storage.load).Load(d_in_block, data);
      // Barrier for smem reuse
      __syncthreads();
      // Compute inclusive prefix sum
      BlockScanT(temp_storage.scan).InclusiveSum(data, data);
      // Barrier for smem reuse
      __syncthreads();
      // Store items from a blocked arrangement
      BlockStoreT(temp_storage.store).Store(d_out_block, data);
    }
}


// Transpose code from:
// https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to
// (BLOCK_THREADS+1)*BLOCK_THREADS.  This pads each row of the 2D block in shared memory
// so that bank conflicts do not occur when threads address the array column-wise.
template <typename T, int BLOCK_THREADS>
__global__ void transpose(int *d_in, int *d_out, int width, int height)
{
	__shared__ T block[BLOCK_THREADS][BLOCK_THREADS+1];

	// read the matrix tile into shared memory
        // load one element per thread from device memory (d_in) and store it
        // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_THREADS + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_THREADS + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = d_in[index_in];
	}

  // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (d_out) in linear order
	xIndex = blockIdx.y * BLOCK_THREADS + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_THREADS + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		d_out[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockScanAlgorithm ALGORITHM>
TGpuTime _compute_intimg_cuda2(const int* image, int* intimg, int w, int h, int grid_size){
  TGpuTime timing;
  auto time_start = std::chrono::steady_clock::now();
  int* image_gpu;
  int* intimg_gpu;
  cudaMalloc(&image_gpu, w*h*sizeof(int));
  cudaMalloc(&intimg_gpu, w*h*sizeof(int));
  auto time_alloc = std::chrono::steady_clock::now();
  timing.alloc = std::chrono::duration_cast<std::chrono::milliseconds>(time_alloc - time_start).count();

  cudaMemcpy(image_gpu, image, w*h*sizeof(int), cudaMemcpyHostToDevice);
  auto time_memcpy = std::chrono::steady_clock::now();
  timing.copy = std::chrono::duration_cast<std::chrono::milliseconds>(time_memcpy - time_alloc).count();

  BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM><<<grid_size, BLOCK_THREADS>>>(
       image_gpu,
       intimg_gpu,
       w, h);

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

  const int transpose_block_threads = 16;
  dim3 grid(h / transpose_block_threads, w / transpose_block_threads, 1);
  dim3 threads(transpose_block_threads, transpose_block_threads, 1);
  transpose<int,transpose_block_threads><<<grid,threads>>>(intimg_gpu, image_gpu, h, w);

  BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM><<<grid_size, BLOCK_THREADS>>>(
       image_gpu,
       intimg_gpu,
       w, h);
  transpose<int,transpose_block_threads><<<grid,threads>>>(intimg_gpu, image_gpu, h, w);
  int* temp = intimg_gpu;
  intimg_gpu = image_gpu;
  image_gpu = temp;

  cudaDeviceSynchronize();

  auto time_end = std::chrono::steady_clock::now();
  timing.compute = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_memcpy).count();

  cudaMemcpy(intimg, intimg_gpu, w*h*sizeof(int), cudaMemcpyDeviceToHost);
  time_memcpy = std::chrono::steady_clock::now();
  timing.copy += std::chrono::duration_cast<std::chrono::milliseconds>(time_memcpy - time_end).count();

  cudaFree(image_gpu);
  cudaFree(intimg_gpu);

  return timing;
}


TGpuTime compute_intimg_cuda2(const int* image, int* intimg, int w, int h, int grid_size){
  switch (w) {
    case 128:
      return _compute_intimg_cuda2<128, 1, BLOCK_SCAN_RAKING>(image, intimg, w, h, grid_size);
      break;
    case 256:
      return _compute_intimg_cuda2<128, 2, BLOCK_SCAN_RAKING>(image, intimg, w, h, grid_size);
      break;
    case 512:
      return _compute_intimg_cuda2<128, 4, BLOCK_SCAN_RAKING>(image, intimg, w, h, grid_size);
      break;
    case 1024:
      return _compute_intimg_cuda2<128, 8, BLOCK_SCAN_RAKING>(image, intimg, w, h, grid_size);
      break;
    case 2048:
      return _compute_intimg_cuda2<128, 16, BLOCK_SCAN_RAKING>(image, intimg, w, h, grid_size);
      break;
    case 4096:
      return _compute_intimg_cuda2<128, 32, BLOCK_SCAN_RAKING>(image, intimg, w, h, grid_size);
      break;
    case 8192:
      return _compute_intimg_cuda2<128, 64, BLOCK_SCAN_RAKING>(image, intimg, w, h, grid_size);
      break;
    default:
      throw std::invalid_argument("Invalid input size for this CUDA implementation.");
  }
}
