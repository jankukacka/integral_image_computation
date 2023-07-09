#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "intimg.h"
#include "intimg_omp.h"
#include "intimg.cuh"
#include "intimg2.cuh"

namespace py = pybind11;

enum Algorithm { cpu, omp, gpu1, gpu2 };

template<Algorithm algorithm>
py::array_t<int> intimg(py::array_t<int> input) {
    py::buffer_info buffer = input.request();

    if (buffer.ndim != 2)
        throw std::runtime_error("Number of dimensions must be one");

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<int>(buffer.shape);

    py::buffer_info buffer_result = result.request();

    int* ptr1 = static_cast<int*>(buffer.ptr);
    int* ptr2 = static_cast<int*>(buffer_result.ptr);

    int grid_size = 5;
    int block_size = 128;
    switch (algorithm) {
      case cpu:
        compute_intimg(ptr1, ptr2, buffer.shape[0], buffer.shape[1]);
        break;
      case omp:
        compute_intimg_parallel(ptr1, ptr2, buffer.shape[0], buffer.shape[1]);
        break;
      case gpu1:
        compute_intimg_cuda1(ptr1, ptr2, buffer.shape[0], buffer.shape[1], grid_size, block_size);
        break;
      case gpu2:
        compute_intimg_cuda2(ptr1, ptr2, buffer.shape[0], buffer.shape[1], grid_size);
        break;
    }

    return result;
}

py::array_t<int> (*fx_cpu)(py::array_t<int>) = intimg<Algorithm::cpu>;
py::array_t<int> (*fx_omp)(py::array_t<int>) = intimg<Algorithm::omp>;
py::array_t<int> (*fx_gpu1)(py::array_t<int>) = intimg<Algorithm::gpu1>;
py::array_t<int> (*fx_gpu2)(py::array_t<int>) = intimg<Algorithm::gpu2>;

PYBIND11_MODULE(intimg, m) {

    m.def("intimg", fx_cpu, "Compute integral image");
    m.def("intimg_parallel", fx_omp, "Compute integral image using multiple threads");
    m.def("intimg_cuda1", fx_omp, "Compute integral image using CUDA");
    m.def("intimg_cuda2", fx_omp, "Compute integral image using fast CUDA");
}
