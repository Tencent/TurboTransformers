#pragma once

namespace fast_transformers {
namespace layers {
namespace kernels {
static constexpr float g_epsilon = 1e-6;

template <typename T>
void LayerNorm(T *input, const T *gamma, const T *beta, const int m,
               const int n) {
#pragma omp parallel for
  for (int batch_idx = 0; batch_idx < m; ++batch_idx) {
    T mean = 0;
    T var = 0;
#pragma omp simd reduction(+ : mean)
    for (int i = batch_idx * n; i < (batch_idx + 1) * n; i++) {
      T t = input[i];
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
      input[j] = beta[i] + gamma[i] * var * (input[j] - mean);
    }
  }
}

} // namespace kernels
} // namespace layers
} // namespace fast_transformers
