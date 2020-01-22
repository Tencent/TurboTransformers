#include "fast_transformers/layers/kernels/activation.h"
#include <numeric>
#include "fast_transformers/core/aligned_scratchpad.h"
#ifdef FT_WITH_CUDA
#include "fast_transformers/core/cuda_device_context.h"
#include "fast_transformers/layers/kernels/gpu_activation_kernel.h"
#endif

namespace fast_transformers {
namespace layers {
namespace kernels {

template <typename T>
static void AddBiasGeLUActKernel(const T* bias, T* out, int64_t batch_size,
                                 int64_t feature_dim) {
  static core::AlignedScratchpad<float> scratchpad;
  float* buff = scratchpad.mutable_data(batch_size * feature_dim);
#pragma omp parallel for
  for (int64_t i = 0; i < batch_size; ++i) {
    int64_t k = 0;
#pragma omp simd
    for (int64_t j = feature_dim * i; j < feature_dim * (i + 1); ++j) {
      float tmp_ = out[j] + bias[k++];
      buff[j] = (0.7978845608028654f * (tmp_ + 0.044715f * tmp_ * tmp_ * tmp_));
    }
    vsTanh(feature_dim, &buff[i * feature_dim], &buff[i * feature_dim]);
    k = 0;
#pragma omp simd
    for (int64_t j = feature_dim * i; j < feature_dim * (i + 1); ++j) {
      out[j] = (out[j] + bias[k++]) * 0.5f * (1.0f + buff[j]);
    }
  }
}

template <typename T>
void AddBiasGeLUAct(const core::Tensor& bias_tensor, core::Tensor* out_tensor) {
  T* out = out_tensor->mutableData<T>();
  const T* bias = bias_tensor.data<T>();

  int64_t m = out_tensor->rows();
  int64_t n = out_tensor->cols();

  if (out_tensor->device_type() == kDLCPU) {
    AddBiasGeLUActKernel(bias, out, m, n);
  } else if (out_tensor->device_type() == kDLGPU) {
#ifdef FT_WITH_CUDA
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUAddBiasGeLUActKernel<T>(bias, out, m, n, cuda_ctx.stream());
#endif
  } else {
    FT_THROW("device_type is not supported");
  }
}

template void AddBiasGeLUAct<float>(const core::Tensor& bias_tensor,
                                    core::Tensor* out_tensor);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
