#include "intimg.h"

float compute_intimg(const int* image, int* intimg, int w, int h){
  auto time_start = std::chrono::steady_clock::now();

  for (int i=0; i<h; i++){
    for (int j=0; j<w; j++){
      int val = 0;
      if (j>0)
        val = intimg[i*w+j-1];
      intimg[i*w+j] = image[i*w+j] + val;
    }
  }

  for (int i=1; i<h; i++){
    for (int j=0; j<w; j++){
      intimg[i*w+j] = intimg[i*w+j] + intimg[(i-1)*w+j];
    }
  }

  auto time_end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

  return duration;
}
