#include "intimg.h"
#include "intimg_omp.h"
#include "intimg.cuh"
#include "intimg2.cuh"

int main(int argc, char const *argv[]) {
  int block_size = 128;
  size_t n = 128*32;
  if (argc >= 2) {
    n = std::stoi(argv[1]);
  }
  int grid_size = 1;
  if (argc >= 3) {
    grid_size = std::stoi(argv[2]);
  }
  size_t repetitions = 10;
  if (argc >= 4) {
    repetitions = std::stoi(argv[3]);
  }

  float* duration = new float[repetitions];
  float* duration_copy = new float[repetitions];
  float* duration_alloc = new float[repetitions];

  int* image = new int[n*n];
  for (int i = 0; i<n*n; i++) {
    image[i] = rand() % 11;
  }

  int* intimg = new int[n*n];
  for (int r=0; r < repetitions; r++) {
    duration[r] = compute_intimg(image, intimg, n, n);
  }
  std::cout << "Computation on CPU took " << mean(duration, repetitions) << "ms (+-" << stdev(duration, repetitions) << ")" << std::endl;

  int* intimg_omp = new int[n*n];
  for (int r=0; r < repetitions; r++) {
    duration[r] = compute_intimg_parallel(image, intimg_omp, n, n);
  }
  std::cout << "Computation on CPU (" << std::getenv("OMP_NUM_THREADS") << " threads) took " << mean(duration, repetitions) << "ms (+-" << stdev(duration, repetitions) << ")" << std::endl;
  check_error(intimg, intimg_omp, n, n);

  int* intimg_cuda1 = new int[n*n];
  for (int r=0; r < repetitions; r++) {
    TGpuTime timing = compute_intimg_cuda1(image, intimg_cuda1, n, n, grid_size, block_size);
    duration[r] = timing.compute;
    duration_copy[r] = timing.copy;
    duration_alloc[r] = timing.alloc;
  }
  std::cout << "Computation on GPU with (" << grid_size << "," << 128 << ") threads took " << mean(duration, repetitions) << "ms(+-" << stdev(duration, repetitions) << ")"; // << std::endl;
  std::cout << " [Memcpy: " << mean(duration_copy, repetitions) << "ms, Alloc: " << mean(duration_alloc, repetitions) << "ms]" << std::endl;
  check_error(intimg, intimg_cuda1, n, n);

  int* intimg_cuda2 = new int[n*n];
  for (int r=0; r < repetitions; r++) {
    TGpuTime timing = compute_intimg_cuda2(image, intimg_cuda2, n, n, grid_size);
    duration[r] = timing.compute;
    duration_copy[r] = timing.copy;
    duration_alloc[r] = timing.alloc;
  }
  std::cout << "Computation on GPU with (" << grid_size << "," << 128 << ") threads took " << mean(duration, repetitions) << "ms(+-" << stdev(duration, repetitions) << ")"; // << std::endl;
  std::cout << " [Memcpy: " << mean(duration_copy, repetitions) << "ms, Alloc: " << mean(duration_alloc, repetitions) << "ms]" << std::endl;
  check_error(intimg, intimg_cuda2, n, n);


  delete [] image;
  delete [] intimg;
  delete [] intimg_omp;
  delete [] intimg_cuda1;
  delete [] intimg_cuda2;
  delete [] duration;
  delete [] duration_copy;
  delete [] duration_alloc;

  return 0;
}
