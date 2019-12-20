#include "fast_transformers/layers/kernels/softmax.h"

#include <immintrin.h>

#include <cmath>
#ifdef FT_WITH_CUDA
#include "fast_transformers/core/cuda_device_context.h"
#include "fast_transformers/layers/kernels/gpu_softmax_kernel.h"
#endif

namespace fast_transformers {
namespace layers {
namespace kernels {
static constexpr float g_epsilon = 1e-6f;
void SoftmaxMask(float* qk_buf, const float* attr_mask, int64_t batch_size,
                 int64_t head_num, int64_t seq_len, float scale) {
  int64_t M = batch_size * head_num * seq_len;
  int64_t N = seq_len;
#pragma omp parallel for
  for (int64_t i = 0; i < M; ++i) {
    auto* qk_buf_ptr = qk_buf + i * N;
    auto attr_mask_offset = i / (head_num * seq_len) * seq_len;
    auto attr_mask_ptr = attr_mask + attr_mask_offset;
#pragma omp simd
    for (int64_t j = 0; j < N; ++j) {
      auto mask_val = attr_mask_ptr[j];
      auto qk_val = qk_buf_ptr[j];
      qk_val = qk_val * scale + mask_val;
      qk_buf_ptr[j] = std::exp(qk_val);
    }
    float sum = 0;
#pragma omp simd reduction(+ : sum)
    for (int64_t j = 0; j < N; ++j) {
      sum += qk_buf_ptr[j];
    }
    auto coef = 1.0f / (sum + g_epsilon);
#pragma omp simd
    for (int64_t j = 0; j < N; ++j) {
      qk_buf_ptr[j] *= coef;
    }
  }
}
void ApplyMaskAndSoftmax(core::Tensor* inout, const core::Tensor& att_mask,
                         float scale) {
  auto batch_size = inout->shape(0);
  auto num_att_heads = inout->shape(1);
  auto seq_len = inout->shape(2);
  if (inout->device_type() == kDLCPU) {
    SoftmaxMask(inout->mutableData<float>(), att_mask.data<float>(), batch_size,
                num_att_heads, seq_len, scale);
  } else if (inout->device_type() == kDLGPU) {
#ifdef FT_WITH_CUDA
    auto& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUSoftmaxMask(inout->mutableData<float>(), att_mask.data<float>(),
                   batch_size, num_att_heads, seq_len, scale,
                   cuda_ctx.stream());
#endif
  } else {
    FT_THROW("device_type is not supported");
  }
}

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
