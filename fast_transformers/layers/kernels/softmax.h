#pragma once
#include <cstdint>

#include "fast_transformers/core/tensor.h"

namespace fast_transformers {
namespace layers {
namespace kernels {
// attr_mask is [batch_size, 1, 1, seq_length] value is 0. / -10000.
template <typename T>
extern void SoftmaxMask(T* qk_buf, const T* attr_mask, int64_t batch_size,
                        int64_t head_num, int64_t seq_len, T scaler);
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
