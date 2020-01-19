#pragma once
#include <stdint.h>

namespace fast_transformers {
namespace layers {
namespace kernels {

template <typename T>
void gpu_sequence(T* data_ptr, int64_t size);

template <typename T>
void gpu_fill(T* data_ptr, int64_t size, T val);

extern void gpu_transform(int64_t* src_data_ptr, float* dst_data_ptr,
                          const int64_t size);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
