#pragma once

namespace fast_transformers {
namespace layers {
namespace kernels {

template <typename T>
void GPUSplitAddBiasTransposeForScore(const T* input_data, const T* bias_data,
                                      T* out_data, int64_t batch_size,
                                      int64_t seq_len, int64_t weight_num,
                                      int64_t num_attention_heads,
                                      int64_t size_per_head,
                                      cudaStream_t stream);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
