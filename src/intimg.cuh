#include <cmath>
#include <chrono>

#include "utils.h"

TGpuTime compute_intimg_cuda1(const int*, int*, int, int, int, int);

__global__
void compute_rowsums(const int* image, int* intimg, int w, int h);

__global__
void compute_colsums(int* intimg, int w, int h);
