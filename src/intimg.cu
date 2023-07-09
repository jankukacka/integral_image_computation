#include "intimg.cuh"

__global__
void compute_rowsums(const int* image, int* intimg, int w, int h) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < h; i += stride) {
    for (int j=0; j<w; j++){
      int val = 0;
      if (j>0)
        val = intimg[i*w+j-1];
      intimg[i*w+j] = image[i*w+j] + val;
    }
  }
}

__global__
void compute_colsums(int* intimg, int w, int h) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i=1; i<w; i++){
    for (int j = index; j < h; j += stride) {
      intimg[i*w+j] = intimg[i*w+j] + intimg[(i-1)*w+j];
    }
  }
}

TGpuTime compute_intimg_cuda1(const int* image, int* intimg, int w, int h, int grid_size, int block_size){
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

  compute_rowsums<<<grid_size,block_size>>>(image_gpu, intimg_gpu, w, h);

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

  compute_colsums<<<grid_size,block_size>>>(intimg_gpu, w, h);

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
