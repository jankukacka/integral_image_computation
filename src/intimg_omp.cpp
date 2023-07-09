#include "intimg_omp.h"


float compute_intimg_parallel(const int* image, int* intimg, int w, int h){
  auto time_start = std::chrono::steady_clock::now();

  #pragma omp parallel
  {
    #pragma omp for
    for (int i=0; i<h; i++){
      for (int j=0; j<w; j++){
        int val = 0;
        if (j>0)
          val = intimg[i*w+j-1];
        intimg[i*w+j] = image[i*w+j] + val;
      }
    }

  // Loop ordering i-j has better cache reuse than j-i, so instead of
  // #pragma omp for
  // we split the work among threads manually
  int col_per_thread = std::ceil(1.0*w / omp_get_num_threads());
  int start = omp_get_thread_num() * col_per_thread;
  int stop = std::min((omp_get_thread_num()+1)*col_per_thread, w);
  for (int i=1; i<h; i++){
    for (int j=start; j<stop; j++){
        intimg[i*w+j] = intimg[(i-1)*w+j] + intimg[i*w+j];
      }
    }
  }

  auto time_end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  return duration;
}
