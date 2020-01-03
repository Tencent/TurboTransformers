#include "fast_transformers/layers/kernels/layer_norm.h"

#ifdef FT_WITH_CUDA
#include "fast_transformers/core/cuda_device_context.h"
#include "fast_transformers/layers/kernels/gpu_layer_norm_kernel.h"
#endif

namespace fast_transformers {
namespace layers {
namespace kernels {
static constexpr float g_epsilon = 1e-12;

template <typename T>
void LayerNorm(const core::Tensor& gamma, const core::Tensor& beta,
               core::Tensor* out_tensor) {
  auto feature_dim = out_tensor->shape(out_tensor->n_dim() - 1);
  int64_t batch_size = std::accumulate(
      &out_tensor->shape(0), &out_tensor->shape(0) + out_tensor->n_dim() - 1, 1,
      std::multiplies<int64_t>());

  auto out = out_tensor->mutableData<T>();

#pragma omp parallel for
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    T mean = static_cast<T>(0);
    T var = static_cast<T>(0);
#pragma omp simd reduction(+ : mean)
    for (int64_t i = batch_idx * feature_dim; i < (batch_idx + 1) * feature_dim;
         i++) {
      T t = out[i];
      mean += t;
      var += t * t;
    }
    mean = mean / feature_dim;
    var = var / feature_dim - mean * mean;

    // 1 / sqrt(var)
    var = 1.f / sqrtf(var + g_epsilon);

    auto beta_ptr = beta.data<T>();
    auto gamma_ptr = gamma.data<T>();
#pragma omp simd
    for (int64_t i = 0; i < feature_dim; ++i) {
      int64_t j = batch_idx * feature_dim + i;
      out[j] = beta_ptr[i] + gamma_ptr[i] * var * (out[j] - mean);
    }
  }
}

template void LayerNorm<float>(const core::Tensor& gamma,
                               const core::Tensor& beta,
                               core::Tensor* out_tensor);

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
  if (input_tensor.device_type() == kDLCPU &&
      bias_tensor.device_type() == kDLCPU &&
      gamma_tensor.device_type() == kDLCPU &&
      beta_tensor.device_type() == kDLCPU &&
      out_tensor->device_type() == kDLCPU) {
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
  } else if (input_tensor.device_type() == kDLGPU &&
             bias_tensor.device_type() == kDLGPU &&
             gamma_tensor.device_type() == kDLGPU &&
             beta_tensor.device_type() == kDLGPU &&
             out_tensor->device_type() == kDLGPU) {
#ifdef FT_WITH_CUDA
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUAddBiasLayerNorm(out, input, bias, gamma, beta, m, n, cuda_ctx.stream());
#endif
  } else {
    FT_THROW("device_type is not supported");
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
