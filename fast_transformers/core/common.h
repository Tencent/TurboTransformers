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

template <typename T>
void ft_seqence(T* data, int64_t size, DLDeviceType device);

template <typename T>
void ft_fill(T* data, int64_t size, T val, DLDeviceType device);

// TODO(jiaruifang): this function should better pass a function in.
// how can we pass a lambda function as __device__ to cuda?
void ft_transform(int64_t* src_data, float* dst_data, int64_t size,
                  DLDeviceType device);

}  // namespace common
}  // namespace core
}  // namespace fast_transformers
