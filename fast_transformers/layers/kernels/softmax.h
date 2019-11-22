#pragma once
#include <stdint.h>

namespace fast_transformers {
namespace layers {
namespace kernels {
// attr_mask is [batch_size, 1, 1, seq_length] value is 0. / -10000.
extern void SoftmaxMask(float* qk_buf, const float* attr_mask,
                        const int64_t batch_size, const int64_t head_num,
                        const int64_t seq_len, float scaler);
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
