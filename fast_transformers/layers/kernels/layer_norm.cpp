#include "fast_transformers/layers/kernels/layer_norm.h"

namespace fast_transformers {
namespace layers {
namespace kernels {
static constexpr float g_epsilon = 1e-12;

template <typename T>
void AddBiasLayerNorm(const core::Tensor& input_tensor,
                      const core::Tensor& bias_tensor,
                      const core::Tensor& gamma_tensor,
                      const core::Tensor& beta_tensor,
                      core::Tensor* out_tensor) {
  T* out = out_tensor->mutableData<T>();
  const T* input = input_tensor.data<T>();
  const T* bias = bias_tensor.data<T>();
  const T* gamma = gamma_tensor.data<T>();
  const T* beta = beta_tensor.data<T>();

  int64_t m = input_tensor.rows();
  int64_t n = input_tensor.cols();
  // TODO(florianzhao): Check the dim of bias_tensor, gamma_tensor, beta_tensor,
  // out_tensor
#pragma omp parallel for
  for (int64_t batch_idx = 0; batch_idx < m; ++batch_idx) {
    float mean = 0;
    float var = 0;
#pragma omp simd reduction(+ : mean)
    for (int64_t i = batch_idx * n; i < (batch_idx + 1) * n; i++) {
      int64_t j = i - batch_idx * n;
      float t = out[i] = out[i] + input[i] + bias[j];
      mean += t;
      var += t * t;
    }
    mean = mean / n;
    var = var / n - mean * mean;

    // 1 / sqrt(var)
    var = 1.f / sqrtf(var + g_epsilon);

#pragma omp simd
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = batch_idx * n + i;
      out[j] = beta[i] + gamma[i] * var * (out[j] - mean);
    }
  }
}

template void AddBiasLayerNorm<float>(const core::Tensor& input_tensor,
                                      const core::Tensor& bias_tensor,
                                      const core::Tensor& gamma_tensor,
                                      const core::Tensor& beta_tensor,
                                      core::Tensor* out_tensor);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
