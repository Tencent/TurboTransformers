#include "fast_transformers/layers/kernels/transpose.h"
#include "fast_transformers/core/common.h"

#include <cstring>
#ifdef FT_WITH_CUDA
#include "fast_transformers/core/cuda_device_context.h"
#include "fast_transformers/layers/kernels/gpu_transpose_kernel.h"
#endif

namespace fast_transformers {
namespace layers {
namespace kernels {

static void TransposeForScoreImpl(float* output, const float* input,
                                  int64_t batch_size, int64_t seq_length,
                                  int64_t num_attention_heads, int64_t width) {
#pragma omp parallel for
  for (int64_t idx = 0; idx < batch_size * seq_length; ++idx) {
    int64_t batch_idx = idx / seq_length;
    int64_t seq_idx = idx % seq_length;
    for (int64_t head_idx = 0; head_idx < num_attention_heads; ++head_idx) {
      auto* src = input +
                  batch_idx * (seq_length * num_attention_heads * width) +
                  seq_idx * width + head_idx * seq_length * width;
      auto* dst = output +
                  batch_idx * (seq_length * num_attention_heads * width) +
                  seq_idx * num_attention_heads * width + head_idx * width;
// std::copy(src, src + width, dst);
#pragma omp simd
      for (int64_t width_idx = 0; width_idx < width; ++width_idx) {
        dst[width_idx] = src[width_idx];
      }
    }
  }
}

void TransposeForScore(core::Tensor* output, const core::Tensor& input) {
  if (input.device_type() == kDLCPU && output->device_type() == kDLCPU) {
    TransposeForScoreImpl(output->mutableData<float>(), input.data<float>(),
                          output->shape(0), output->shape(1), input.shape(1),
                          input.shape(3));
  } else if (input.device_type() == kDLGPU && output->device_type() == kDLGPU) {
#ifdef FT_WITH_CUDA
    auto batch_size = output->shape(0);
    auto seq_length = output->shape(1);
    auto num_attention_heads = input.shape(1);
    auto width = input.shape(3);
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUTransposeForScore<float>(
        input.data<float>(), output->mutableData<float>(), batch_size,
        seq_length, num_attention_heads, width, cuda_ctx.stream());
#endif
  } else {
    FT_THROW("device_type is not supported");
  }
}

void SplitAddBiasTransposeForScore(core::Tensor* output_tensor,
                                   const core::Tensor& input_tensor,
                                   const core::Tensor& bias_tensor) {
  FT_ENFORCE_EQ(output_tensor->n_dim(), 5,
                "output_tensor should be (weight_num, batch_size, seq_length, "
                "num_attention_heads, size_per_head)");
  FT_ENFORCE_EQ(output_tensor->shape(0), 3,
                "output_tensor should be (3, batch_size, seq_length, "
                "num_attention_heads, size_per_head)");

  auto batch_size = output_tensor->shape(1);
  auto seq_length = output_tensor->shape(3);
  auto weight_num = output_tensor->shape(0);
  auto num_attention_heads = output_tensor->shape(2);
  auto width = output_tensor->shape(4);
  auto input = input_tensor.data<float>();
  auto bias = bias_tensor.data<float>();
  auto output = output_tensor->mutableData<float>();

  FT_ENFORCE_EQ(core::common::is_same_device_ctx(input_tensor.device_ctx(),
                                                 bias_tensor.device_ctx()),
                true,
                "SplitAddBiasTransposeForScore: input_tensor and bias_tensor "
                "should have the same device type and device id.");
  FT_ENFORCE_EQ(core::common::is_same_device_ctx(input_tensor.device_ctx(),
                                                 output_tensor->device_ctx()),
                true,
                "SplitAddBiasTransposeForScore: input_tensor and output_tensor "
                "should have the same device type and device id.");

  if (output_tensor->device_type() == kDLCPU &&
      input_tensor.device_type() == kDLCPU &&
      bias_tensor.device_type() == kDLCPU) {
#pragma omp parallel for
    for (int64_t idx = 0; idx < batch_size * weight_num * seq_length; ++idx) {
      auto batch_idx = idx / (seq_length * weight_num);
      auto seq_idx = idx / weight_num % seq_length;
      auto weight_idx = idx % weight_num;

      for (int64_t head_idx = 0; head_idx < num_attention_heads; ++head_idx) {
        auto* src_ptr =
            input +
            batch_idx *
                (seq_length * weight_num * num_attention_heads * width) +
            seq_idx * weight_num * num_attention_heads * width +
            weight_idx * (num_attention_heads * width) + head_idx * width;
        auto* dst_ptr = output +
                        weight_idx * (batch_size * num_attention_heads *
                                      seq_length * width) +
                        batch_idx * (num_attention_heads * seq_length * width) +
                        head_idx * seq_length * width + seq_idx * width;
        auto* bias_ptr =
            bias + weight_idx * width * num_attention_heads + head_idx * width;
#pragma omp simd
        for (int64_t width_idx = 0; width_idx < width; ++width_idx) {
          dst_ptr[width_idx] = src_ptr[width_idx] + bias_ptr[width_idx];
        }
      }
    }  // end for
  } else if (output_tensor->device_type() == kDLGPU &&
             input_tensor.device_type() == kDLGPU &&
             bias_tensor.device_type() == kDLGPU) {
#ifdef FT_WITH_CUDA
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUSplitAddBiasTransposeForScore<float>(
        input, bias, output, batch_size, seq_length, weight_num,
        num_attention_heads, width, cuda_ctx.stream());
#endif
  } else {
    FT_THROW("device_type is not supported");
  }
}

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
