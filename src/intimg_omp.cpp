#include "intimg_omp.h"


// int main(int argc, char const *argv[]) {
//   const size_t repetitions = 10;
//   size_t n = 16;
//   if (argc >= 2) {
//     n = std::stoi(argv[1]);
//   }
//
//   int* image = new int[n*n];
//   for (int i = 0; i<n*n; i++) {
//     image[i] = rand() % 11;
//   }
//
//   if (n<6) {
//     std::cout << "Image" << '\n';
//     print_array(image, n, n);
//     std::cout << '\n';
//   }
//
//   float duration[repetitions];
//   int* intimg = new int[n*n];
//   for (int r=0;r<repetitions;r++) {
//     duration[r] = compute_intimg(image, intimg, n, n);
//   }
//   if (n<6) {
//     std::cout << "Integral image (serial)" << '\n';
//     print_array(intimg, n, n);
//     std::cout << '\n';
//   }
//   std::cout << "Computation on CPU took " << mean(duration, repetitions) << "ms (+-" << stdev(duration, repetitions) << ")" << std::endl;
//
//
//   int* intimg2 = new int[n*n];
//   for (int r=0;r<repetitions;r++) {
//     duration[r] = compute_intimg_parallel(image, intimg2, n, n);
//   }
//   if (n<6) {
//     std::cout << "Integral image (parallel)" << '\n';
//     print_array(intimg2, n, n);
//     std::cout << '\n';
//   }
//   std::cout << "Computation on CPU (" << std::getenv("OMP_NUM_THREADS") << " threads) took " << mean(duration, repetitions) << "ms (+-" << stdev(duration, repetitions) << ")" << std::endl;
//   check_error(intimg, intimg2, n, n);
//
//   int* intimg3 = new int[n*n];
//   for (int r=0;r<repetitions;r++) {
//     duration[r] = compute_intimg_parallel2(image, intimg3, n, n);
//   }
//   std::cout << "Computation on CPU (" << std::getenv("OMP_NUM_THREADS") << " threads) took " << mean(duration, repetitions) << "ms (+-" << stdev(duration, repetitions) << ")" << std::endl;
//   check_error(intimg, intimg3, n, n);
//
//   delete [] image;
//   delete [] intimg;
//   delete [] intimg2;
//
//   return 0;
// }

// float compute_intimg_parallel(const int* image, int* intimg, int w, int h){
//   auto time_start = std::chrono::steady_clock::now();
//
//   #pragma omp parallel
//   {
//     #pragma omp for
//     for (int i=0; i<h; i++){
//       for (int j=0; j<w; j++){
//         int val = 0;
//         if (j>0)
//         val = intimg[i*w+j-1];
//       intimg[i*w+j] = image[i*w+j] + val;
//       }
//     }
//
//   #pragma omp for
//   for (int j=0; j<w; j++){
//     for (int i=0; i<h; i++){
//         // int val = 0;
//         if (i>0)
//         //   val = intimg[(i-1)*w+j];
//           intimg[i*w+j] = intimg[i*w+j] + intimg[(i-1)*w+j];
//       }
//     }
//   }
//
//   auto time_end = std::chrono::steady_clock::now();
//   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
//   return duration;
// }

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
