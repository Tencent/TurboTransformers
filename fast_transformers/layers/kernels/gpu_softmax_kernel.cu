#include <cuda_runtime.h>
#include <numeric>
#include "fast_transformers/layers/kernels/gpu_common.h"
#include "fast_transformers/layers/kernels/gpu_softmax_kernel.h"

// copy from
// https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer

namespace fast_transformers {
namespace layers {
namespace kernels {

__global__ void softmax_kernel(float* qk_buf_, const float* attr_mask,
                               int batch_size, int head_num, int seq_len,
                               float scaler) {
  int batch_id = blockIdx.x / head_num;
  int qk_offset = blockIdx.x * seq_len * seq_len;

  __shared__ float s_sum, s_max;

  for (int i = 0; i < seq_len; ++i) {
    float qk =
        threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    float mask_val =
        threadIdx.x < seq_len
            ? (float)attr_mask[threadIdx.x % seq_len + batch_id * seq_len]
            : 0.0f;

    // mask_val = (1.0f - mask_val) * -10000.0f;
    float tmp =
        threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val) : -1e20f;

    float max_val = blockReduceMax(tmp);

    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();

    qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

    float sum_val = blockReduceSum(qk);

    if (threadIdx.x == 0) {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if (threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = (float)(qk / s_sum);

    qk_offset += seq_len;
  }
}

__global__ void softmax_kernel_v2(float* qk_buf_, const float* attr_mask,
                                  const int batch_size, const int head_num,
                                  const int seq_len, const float scaler) {
  int batch_id = blockIdx.x / head_num / seq_len;
  int qk_offset = blockIdx.x * seq_len;
  int mask_offset = batch_id * seq_len;

  __shared__ float s_sum, s_max;

  float qk =
      threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
  // float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x +
  // mask_offset] :0.0f;
  float mask_val = threadIdx.x < seq_len
                       ? (float)attr_mask[threadIdx.x + mask_offset]
                       : 0.0f;

  // mask_val = (1.0f - mask_val) * -10000.0f;

  float tmp =
      threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val) : -1e20f;
  float max_val = blockReduceMax(tmp);
  if (threadIdx.x == 0) s_max = max_val;
  __syncthreads();

  float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
  float sum_val = blockReduceSum(qk_tmp);

  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-6f;
  }
  __syncthreads();

  if (threadIdx.x < seq_len)
    qk_buf_[threadIdx.x + qk_offset] = (float)(qk_tmp / s_sum);
}

template <>
void GPUSoftmaxMask(float* qk_buf, const float* attr_mask, int64_t batch_size,
                    int64_t head_num, int64_t seq_len, float scale,
                    cudaStream_t stream) {
  dim3 block, grid;
  if (seq_len <= 32)
    block.x = 32;
  else if (seq_len > 32 && seq_len <= 64)
    block.x = 64;
  else if (seq_len > 64 && seq_len <= 128)
    block.x = 128;
  else if (seq_len > 128 && seq_len <= 256)
    block.x = 256;
  else if (seq_len > 256 && seq_len <= 512)
    block.x = 512;
  else
    block.x = 1024;
  if (batch_size * head_num <= 120) {
    grid.x = batch_size * head_num * seq_len;
    softmax_kernel_v2<<<grid, block, 0, stream>>>(qk_buf, attr_mask, batch_size,
                                                  head_num, seq_len, scale);
  } else {
    grid.x = batch_size * head_num;
    softmax_kernel<<<grid, block, 0, stream>>>(qk_buf, attr_mask, batch_size,
                                               head_num, seq_len, scale);
  }
}

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
