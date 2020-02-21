#include <cuda_runtime.h>
#include <immintrin.h>
#include <numeric>
#include "fast_transformers/layers/kernels/gpu_common.h"
#include "fast_transformers/layers/kernels/gpu_layer_norm_kernel.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

template <bool isAdd>
static __global__ void layer_norm_kernel(float* out, const float* input,
                                         const float* bias, const float* gamma,
                                         const float* beta, int m, int n) {
  int tid = threadIdx.x;
  int offset = blockIdx.x * n + tid;
  __shared__ float s_mean;
  __shared__ float s_variance;

  float local_out = 0.0f;
  if (isAdd) {
    local_out = out[offset] + input[offset] + __ldg(&bias[tid]);
  } else {
    local_out = out[offset];
  }

  float sum1 = local_out, sum2 = local_out * local_out;
  blockReduceSum_Elem2(&sum1, &sum2);

  if (tid == 0) {
    float mean = sum1 / n;
    float mean_2 = sum2 / n;
    s_mean = mean;
    s_variance = rsqrtf(mean_2 - mean * mean + 1e-6f);
  }
  __syncthreads();

  out[offset] = (local_out - s_mean) * s_variance * __ldg(&gamma[tid]) +
                __ldg(&beta[tid]);
}

template <>
void GPUAddBiasLayerNorm(float* out, const float* input, const float* bias,
                         const float* gamma, const float* beta, int m, int n,
                         cudaStream_t stream) {
  dim3 block(n);
  if (block.x > 1024) {
    throw std::runtime_error(
        "GPUAddBiasLayerNorm thread block size large than 1024");
  }
  dim3 grid(m);
  layer_norm_kernel<true>
      <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}

template <>
void GPULayerNorm(float* out, const float* gamma, const float* beta, int m,
                  int n, cudaStream_t stream) {
  dim3 block(n);
  if (block.x > 1024) {
    throw std::runtime_error(
        "GPUAddBiasLayerNorm thread block size large than 1024");
  }
  float* dummy = nullptr;

  dim3 grid(m);
  layer_norm_kernel<false>
      <<<grid, block, 0, stream>>>(out, out, dummy, gamma, beta, m, n);
}

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
