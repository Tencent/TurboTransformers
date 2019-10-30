#pragma once
#include "fast_transformers/core/tensor.h"
#include <numeric>

namespace fast_transformers {
namespace layers {
namespace kernels {
static constexpr float g_epsilon = 1e-6;

template <typename T>
void LayerNorm(core::Tensor &out_tensor, const core::Tensor &gamma,
               const core::Tensor &beta) {
  auto feature_dim = out_tensor.shape(out_tensor.n_dim() - 1);
  int64_t batch_size = std::accumulate(
      &out_tensor.shape(0), &out_tensor.shape(0) + out_tensor.n_dim() - 1, 1,
      std::multiplies<int64_t>());

  auto out = out_tensor.mutableData<T>();

#pragma omp parallel for
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    T mean = 0;
    T var = 0;
#pragma omp simd reduction(+ : mean)
    for (int i = batch_idx * feature_dim; i < (batch_idx + 1) * feature_dim;
         i++) {
      T t = out[i];
      mean += t;
      var += t * t;
    }
    mean = mean / feature_dim;
    var = var / feature_dim - mean * mean;

    // 1 / sqrt(var)
    var = 1.f / sqrtf(var + g_epsilon);

#pragma omp simd
    auto beta_ptr = beta.data<float>();
    auto gamma_ptr = gamma.data<float>();
    for (int i = 0; i < feature_dim; ++i) {
      int j = batch_idx * feature_dim + i;
      out[j] = beta_ptr[i] + gamma_ptr[i] * var * (out[j] - mean);
    }
  }
}

} // namespace kernels
} // namespace layers
} // namespace fast_transformers
