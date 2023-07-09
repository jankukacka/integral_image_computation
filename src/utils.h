#include <iostream>
#include <chrono>
#include <omp.h>
#include <cmath>

#ifndef __UTILS_INCLUDE
#define __UTILS_INCLUDE
struct TGpuTime {
  float alloc;
  float copy;
  float compute;
};
#endif

float mean(const float* array, size_t n);
float stdev(const float* array, size_t n);
void print_array(int* array, int h, int w);
void check_error(const int* array1, const int* array2, int h, int w);
