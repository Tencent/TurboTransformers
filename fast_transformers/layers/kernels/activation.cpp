#include "fast_transformers/layers/kernels/activation.h"
#include "fast_transformers/core/device_context.h"

#include <immintrin.h>

#include <numeric>

#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/eigen-tensor.h"
#ifdef WITH_CUDA
#include "fast_transformers/layers/kernels/gpu_activation_kernel.h"
#endif

namespace fast_transformers {
namespace layers {
namespace kernels {

template <typename T>
void AddBiasGeLUAct(const core::Tensor& bias_tensor, core::Tensor* out_tensor) {
  T* out = out_tensor->mutableData<T>();
  const T* bias = bias_tensor.data<T>();

  int64_t m = out_tensor->rows();
  int64_t n = out_tensor->cols();

  if(out_tensor->device_type() == kDLCPU) {
    static core::AlignedScratchpad<float> scratchpad;
    float* buff = scratchpad.mutable_data(n * m);
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
  } else if (out_tensor->device_type() == kDLGPU) {
#ifdef WITH_CUDA
    core::DeviceContextPool& pool = core::DeviceContextPool::Instance();
    core::CUDADeviceContext* gpu_ctx = static_cast<core::CUDADeviceContext *>(pool.Get(kDLGPU));
    GPUAddBiasGeLUActKernel<T>(bias, out, m, n, gpu_ctx->stream());
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
