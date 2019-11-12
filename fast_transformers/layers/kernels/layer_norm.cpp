#include "fast_transformers/layers/kernels/layer_norm.h"

namespace fast_transformers {
namespace layers {
namespace kernels {
static constexpr float g_epsilon = 1e-12;

void LayerNorm(core::Tensor& out_tensor, const core::Tensor& gamma,
               const core::Tensor& beta) {
  auto feature_dim = out_tensor.shape(out_tensor.n_dim() - 1);
  int64_t batch_size = std::accumulate(
      &out_tensor.shape(0), &out_tensor.shape(0) + out_tensor.n_dim() - 1, 1,
      std::multiplies<int64_t>());

  auto out = out_tensor.mutableData<float>();

#pragma omp parallel for
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    float mean = 0;
    float var = 0;
#pragma omp simd reduction(+ : mean)
    for (int i = batch_idx * feature_dim; i < (batch_idx + 1) * feature_dim;
         i++) {
      float t = out[i];
      mean += t;
      var += t * t;
    }
    mean = mean / feature_dim;
    var = var / feature_dim - mean * mean;

    // 1 / sqrt(var)
    var = 1.f / sqrtf(var + g_epsilon);

    auto beta_ptr = beta.data<float>();
    auto gamma_ptr = gamma.data<float>();
#pragma omp simd
    for (int i = 0; i < feature_dim; ++i) {
      int j = batch_idx * feature_dim + i;
      out[j] = beta_ptr[i] + gamma_ptr[i] * var * (out[j] - mean);
    }
  }
}

void AddBiasLayerNorm(float* out, const float* input, const float* bias,
                      const float* gamma, const float* beta, int m, int n) {
#pragma omp parallel for
  for (int batch_idx = 0; batch_idx < m; ++batch_idx) {
    float mean = 0;
    float var = 0;
#pragma omp simd reduction(+ : mean)
    for (int i = batch_idx * n; i < (batch_idx + 1) * n; i++) {
      int j = i - batch_idx * n;
      float t = out[i] = out[i] + input[i] + bias[j];
      mean += t;
      var += t * t;
    }
    mean = mean / n;
    var = var / n - mean * mean;

    // 1 / sqrt(var)
    var = 1.f / sqrtf(var + g_epsilon);

#pragma omp simd
    for (int i = 0; i < n; ++i) {
      int j = batch_idx * n + i;
      out[j] = beta[i] + gamma[i] * var * (out[j] - mean);
    }
  }
}

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
