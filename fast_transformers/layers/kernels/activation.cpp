#include "fast_transformers/layers/kernels/activation.h"
#include <immintrin.h>
#include <numeric>
#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/eigen-tensor.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

void AddBiasGeLUAct(float *out, const float *bias, int64_t m, int64_t n) {
  static core::AlignedScratchpad<float> scratchpad;
  float *buff = scratchpad.mutable_data(n * m);
#ifdef __USE_INTEL_COMPILER__
#pragma omp parallel for
  for (int64_t i = 0; i < m; ++i) {
    int64_t k = 0;
    __m256 const_1 = _mm256_set1_ps(0.7978845608028654);
    __m256 const_2 = _mm256_set1_ps(0.044715);
    __m256 const_3 = _mm256_set1_ps(0.5);
    __m256 const_4 = _mm256_set1_ps(1.0);

#pragma unroll(4)
    for (int64_t j = n * i; j < n * (i + 1); j += 8) {
      __m256 out_vect = _mm256_load_ps(out + j);
      __m256 bias_vect = _mm256_load_ps(bias + k);
      __m256 tmp_ = out_vect + bias_vect;
      __m256 tmp2_ =
          _mm256_tanh_ps(const_1 * (tmp_ + const_2 * tmp_ * tmp_ * tmp_));
      out_vect = (tmp_)*const_3 * (const_4 + tmp2_);
      _mm256_store_ps(out + j, out_vect);
      k += 8;
    }
  }
#else
#pragma omp parallel for
  for (int64_t i = 0; i < m; ++i) {
    int64_t k = 0;
#pragma omp simd
    for (int64_t j = n * i; j < n * (i + 1); ++j) {
      float tmp_ = out[j] + bias[k++];
      buff[j] = (0.7978845608028654f * (tmp_ + 0.044715f * tmp_ * tmp_ * tmp_));
    }
    vsTanh(n, &buff[i * n], &buff[i * n]);
    k = 0;
#pragma omp simd
    for (int64_t j = n * i; j < n * (i + 1); ++j) {
      out[j] = (out[j] + bias[k++]) * 0.5f * (1.0f + buff[j]);
    }
  }
#endif
}
void AddBiasGeLUAct(const core::Tensor &bias, core::Tensor *inout) {
  int m = inout->rows();
  int n = inout->cols();
  core::EigenFloatTensor<2> in_mat(inout->mutableData<float>(), m, n);
  auto bias_vec = core::to_tensor<1>(bias);

  auto before_act =
      (in_mat + bias_vec.broadcast(Eigen::DSizes<int, 2>(m, 1))).eval();

  in_mat.device(core::CPUDevice()) =
      before_act * 0.5f *
      (1.0f + (0.7978845608028654f *
               (before_act + 0.044715f * before_act * before_act * before_act))
                  .tanh());
}

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
