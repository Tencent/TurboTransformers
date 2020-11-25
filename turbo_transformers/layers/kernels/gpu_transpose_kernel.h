

#pragma once

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <typename T>
void GPUSplitAddBiasTransposeForScore(const T* input_data, const T* bias_data,
                                      T* out_data, int64_t batch_size,
                                      int64_t seq_len, int64_t weight_num,
                                      int64_t num_attention_heads,
                                      int64_t size_per_head,
                                      cudaStream_t stream);

template <typename T>
void GPUSplitAddBiasTransposeForScoreThreeOutput(
    const T* input_data, const T* bias_data, int64_t batch_size,
    int64_t seq_len, int64_t weight_num, int64_t num_attention_heads,
    int64_t size_per_head, cudaStream_t stream, T* q_out_data, T* k_out_data,
    T* v_out_data);

template <typename T, bool AddBias>
void GPUTransposeForScore(const T* input_data, const T* bias,
                          int64_t batch_size, int64_t seq_len,
                          int64_t num_attention_heads, int64_t size_per_head,
                          cudaStream_t stream, T* output_data);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
