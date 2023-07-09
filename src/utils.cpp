#include "utils.h"

float mean(const float* array, size_t n) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum += array[i];
  }
  return sum/n;
}

float stdev(const float* array, size_t n) {
  float avg = mean(array, n);
  double diff_sum = 0;
  for (int i = 0; i < n; i++) {
    float diff = (array[i]-avg);
    diff_sum += diff*diff;
  }
  return std::sqrt(diff_sum/n);
}


void print_array(int* array, int h, int w) {
  for (int i=0; i<h; i++){
    for (int j=0; j<w; j++){
      if ((i <= 5 || i >= h-5)
          && (j <= 5 || j >= w-5)) {
        std::cout << array[i*w+j] << ' ';
      }
    }
    if (i <= 5 || i >= h-5)
      std::cout << '\n';
  }
}

void check_error(const int* array1, const int* array2, int h, int w) {
  /// Check correctness
  int max_error = 0;
  int error_index = 0;
  for (int i = 0; i < h*w; i++) {
    int error = std::abs(array1[i]-array2[i]);
    if (error > max_error)
      max_error = error;
      error_index = i;
  }

  if (max_error > 0) {
    std::cout << "Max error = " << max_error;
    std::cout << " (" << error_index << ")" << std::endl;
  } else {
    // std::cout << "Correct result!" << std::endl;
  }
}
