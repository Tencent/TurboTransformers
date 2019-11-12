#include "fast_transformers/layers/kernels/softmax.h"
#include <immintrin.h>
#include <cmath>
#include "fast_transformers/core/blas.h"

namespace fast_transformers {
namespace layers {
namespace kernels {
static constexpr float g_epsilon = 1e-6f;
void SoftmaxMask(float* qk_buf, const float* attr_mask,
                 const int64_t batch_size, const int64_t head_num,
                 const int64_t seq_len, float scaler) {
  int64_t M = batch_size * head_num * seq_len;
  int64_t N = seq_len;
#ifdef __USE_INTEL_COMPILER__
#pragma omp parallel for
  for (int64_t i = 0; i < M; ++i) {
    int64_t attr_mask_offset = i / (head_num * seq_len) * seq_len;
    for (int64_t j = 0; j < N; j += 8) {
      __m256 mask_val_vect = _mm256_load_ps(attr_mask + attr_mask_offset + j);
      __m256 qk_val_vect = _mm256_load_ps(qk_buf + i * N + j);
      qk_val_vect = qk_val_vect * _mm256_set1_ps(scaler) + mask_val_vect;
      mask_val_vect = _mm256_exp_ps(qk_val_vect);
      _mm256_store_ps(qk_buf + i * N + j, mask_val_vect);
    }

    float sum = 0.;
#pragma omp simd reduction(+ : sum)
    for (int64_t j = 0; j < N; ++j) {
      sum += qk_buf[i * N + j];
    }
    core::Blas().sscal_(N, 1.0f / (sum + 1e-6f), &qk_buf[i * N], 1);
  }
}
#else
#pragma omp parallel for
  for (int64_t i = 0; i < M; ++i) {
#pragma omp simd
    for (int64_t j = 0; j < N; ++j) {
      // float mask_val =
      //     1. * (attr_mask[i / (head_num * seq_len) * seq_len * seq_len +
      //                     i % seq_len * N + j]);
      // mask_val = (1.0f - mask_val) * (-10000.0f);
      float mask_val = attr_mask[i / (head_num * seq_len) * seq_len + j];
      float qk_val = qk_buf[i * N + j];
      qk_val = qk_val * scaler + mask_val;
      // TODO: use MKL will be fast vsExp(N, &qk_buf[i*N], &qk_buf[i*N]);
      qk_buf[i * N + j] = std::exp(qk_val);
    }
    float sum = 0.;
#pragma omp simd reduction(+ : sum)
    for (int64_t j = 0; j < N; ++j) {
      sum += qk_buf[i * N + j];
    }
    core::Blas().sscal_(N, 1.0f / (sum + g_epsilon), &qk_buf[i * N], 1);
  }
#endif
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
}  // namespace fast_transformers
