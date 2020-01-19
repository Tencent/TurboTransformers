#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "fast_transformers/layers/kernels/gpu_utils.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

template <typename T>
void gpu_sequence(T* data_ptr, int64_t size) {
  thrust::device_ptr<T> data_dev_ptr = thrust::device_pointer_cast(data_ptr);
  thrust::sequence(thrust::device, data_dev_ptr, data_dev_ptr + size);
}

template void gpu_sequence<int64_t>(int64_t* data_ptr, int64_t size);
template void gpu_sequence<float>(float* data_ptr, int64_t size);

template <typename T>
void gpu_fill(T* data_ptr, int64_t size, T val) {
  thrust::device_ptr<T> data_dev_ptr = thrust::device_pointer_cast(data_ptr);
  thrust::fill(thrust::device, data_dev_ptr, data_dev_ptr + size, val);
}

template void gpu_fill<int64_t>(int64_t* data_ptr, int64_t size, int64_t val);
template void gpu_fill<float>(float* data_ptr, int64_t size, float val);

struct negative_functor {
  __host__ __device__ float operator()(const int64_t& v) const {
    return -10000.0f * (1 - v);
  }
};
void gpu_transform(int64_t* src_data_ptr, float* dst_data_ptr,
                   const int64_t size) {
  negative_functor func;
  thrust::device_ptr<int64_t> src_data_ptr_dev_ptr =
      thrust::device_pointer_cast(src_data_ptr);
  thrust::device_ptr<float> dst_data_ptr_dev_ptr =
      thrust::device_pointer_cast(dst_data_ptr);
  thrust::transform(src_data_ptr_dev_ptr, src_data_ptr_dev_ptr + size,
                    dst_data_ptr_dev_ptr, func);
}

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
