#pragma once
#include <dlpack/dlpack.h>
#include "fast_transformers/core/tensor.h"

#ifdef FT_WITH_CUDA
#include "fast_transformers/layers/kernels/gpu_utils.h"
#endif

namespace fast_transformers {
namespace core {
namespace common {

extern bool is_same_device_ctx(DLContext t1, DLContext t2);

extern bool is_same_shape(const Tensor& t1, const Tensor& t2);

template <DLDeviceType device, typename T>
void ft_seqence(T* data, int64_t size) {
  if (device == kDLCPU) {
    std::iota(data, data + size, static_cast<T>(0));
  } else if (device == kDLGPU) {
#ifdef FT_WITH_CUDA
    fast_transformers::layers::kernels::gpu_sequence(data, size);
#else
    FT_THROW("code is not compiled with CUDA.");
#endif
  } else {
    FT_THROW("device_type is not supported");
  }
}

// FIXME(jiaruifang) How to move the following 3 functions to a .cpp file?
template <DLDeviceType device, typename T>
void ft_fill(T* data, int64_t size, T val) {
  if (device == kDLCPU) {
    std::fill(data, data + size, val);
  } else if (device == kDLGPU) {
#ifdef FT_WITH_CUDA
    layers::kernels::gpu_fill(data, size, val);
#else
    FT_THROW("code is not compiled with CUDA.");
#endif
  } else {
    FT_THROW("device_type is not supported");
  }
}

// TODO(jiaruifang): this function should better pass a function in.
// how can we pass a lambda function as __device__ to cuda?
template <DLDeviceType device>
void ft_transform(int64_t* src_data, float* dst_data, int64_t size) {
  if (device == kDLCPU) {
    std::transform(src_data, src_data + size, dst_data,
                   [](int64_t v) { return -10000.0f * (1 - v); });
  } else if (device == kDLGPU) {
#ifdef FT_WITH_CUDA
    layers::kernels::gpu_transform(src_data, dst_data, size);
#else
    FT_THROW("code is not compiled with CUDA.");
#endif
  } else {
    FT_THROW("device_type is not supported");
  }
}

}  // namespace common
}  // namespace core
}  // namespace fast_transformers
