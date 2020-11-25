

#pragma once

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <typename T>
void GPUSoftmaxMask(T* qk_buf, const T* attr_mask, int64_t batch_size,
                    int64_t head_num, int64_t from_seq_len, int64_t to_seq_len,
                    float scale, bool is_2D, cudaStream_t stream);
}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
