#pragma once

namespace fast_transformers {
namespace layers {
namespace kernels {

void GPUSoftmaxMask(float* qk_buf, const float* attr_mask, int64_t batch_size,
                    int64_t head_num, int64_t seq_len, float scale,
                    cudaStream_t stream);
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
