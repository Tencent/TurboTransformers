#pragma once

namespace fast_transformers {
namespace layers {
namespace kernels {

template <typename T>
void GPUSoftmaxMask(T* qk_buf, const T* attr_mask, int64_t batch_size,
                    int64_t head_num, int64_t seq_len, float scale,
                    cudaStream_t stream);
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
