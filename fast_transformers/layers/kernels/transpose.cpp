#include "fast_transformers/layers/kernels/transpose.h"

#include <cstring>

namespace fast_transformers {
namespace layers {
namespace kernels {

template <typename T>
void TransposeForScore(T* output, const T* input,
                       const std::vector<int64_t>& shape) {
  auto batch_size = shape[0];
  auto seq_length = shape[1];
  auto num_attention_heads = shape[2];
  auto width = shape[3];
#pragma omp parallel for
  for (int64_t idx = 0; idx < batch_size * seq_length; ++idx) {
    int64_t batch_idx = idx / seq_length;
    int64_t seq_idx = idx % seq_length;
    for (int64_t head_idx = 0; head_idx < num_attention_heads; ++head_idx) {
      const T* src = input +
                     batch_idx * (seq_length * num_attention_heads * width) +
                     seq_idx * num_attention_heads * width + head_idx * width;
      T* dst = output + batch_idx * (seq_length * num_attention_heads * width) +
               seq_idx * width + head_idx * seq_length * width;
// std::copy(src, src + width, dst);
#pragma omp simd
      for (int64_t width_idx = 0; width_idx < width; ++width_idx) {
        dst[width_idx] = src[width_idx];
      }
    }
  }
}

template <typename T>
void AdBiasTransposeForScore(T* output, const T* input, const T* bias,
                             const std::vector<int64_t>& shape) {
  auto batch_size = shape[0];
  auto seq_length = shape[1];
  auto num_attention_heads = shape[2];
  auto width = shape[3];
// TODO if batch_size * seq_length is not larger than thread_num, we can
// parallel on batch_size * seq_length * num_attention_heads
#pragma omp parallel for
  for (int64_t idx = 0; idx < batch_size * seq_length; ++idx) {
    auto batch_idx = idx / seq_length;  // batch idx
    auto seq_idx = idx % seq_length;    // seq_length idx
    for (int64_t head_idx = 0; head_idx < num_attention_heads; ++head_idx) {
      const T* src = input +
                     batch_idx * (seq_length * num_attention_heads * width) +
                     seq_idx * num_attention_heads * width + head_idx * width;
      T* dst = output + batch_idx * (seq_length * num_attention_heads * width) +
               seq_idx * width + head_idx * seq_length * width;
      const T* src_bias = bias + head_idx * width;
#pragma omp simd
      for (int64_t width_idx = 0; width_idx < width; ++width_idx) {
        dst[width_idx] = src[width_idx] + src_bias[width_idx];
      }
    }
  }  // end for
}

template <typename T>
void SplitAddbiasTransposeForScore(T* output, const T* input, const T* bias,
                                   const std::vector<int64_t>& shape) {
  auto batch_size = shape[0];
  auto seq_length = shape[1];
  auto weight_num = shape[2];
  auto num_attention_heads = shape[3];
  auto width = shape[4];

#pragma omp parallel for
  for (int64_t idx = 0; idx < batch_size * weight_num * seq_length; ++idx) {
    auto batch_idx = idx / (seq_length * weight_num);
    auto seq_idx = idx / weight_num % seq_length;
    auto weight_idx = idx % weight_num;

    for (int64_t head_idx = 0; head_idx < num_attention_heads; ++head_idx) {
      const T* src_ptr =
          input +
          batch_idx * (seq_length * weight_num * num_attention_heads * width) +
          seq_idx * weight_num * num_attention_heads * width +
          weight_idx * (num_attention_heads * width) + head_idx * width;
      T* dst_ptr =
          output +
          weight_idx * (batch_size * num_attention_heads * seq_length * width) +
          batch_idx * (num_attention_heads * seq_length * width) +
          head_idx * seq_length * width + seq_idx * width;
      const T* bias_ptr =
          bias + weight_idx * width * num_attention_heads + head_idx * width;
#pragma omp simd
      for (int64_t width_idx = 0; width_idx < width; ++width_idx) {
        dst_ptr[width_idx] = src_ptr[width_idx] + bias_ptr[width_idx];
      }
    }
  }  // end for
}

template void TransposeForScore<float>(float* output, const float* input,
                                       const std::vector<int64_t>& shape);
template void AdBiasTransposeForScore<float>(float* output, const float* input,
                                             const float* bias,
                                             const std::vector<int64_t>& shape);
template void SplitAddbiasTransposeForScore<float>(
    float* output, const float* input, const float* bias,
    const std::vector<int64_t>& shape);
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
