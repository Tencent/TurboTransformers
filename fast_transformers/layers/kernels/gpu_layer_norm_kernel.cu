#include "fast_transformers/layers/kernels/gpu_layer_norm_kernel.h"
#include "fast_transformers/layers/kernels/gpu_common.h"
#include <immintrin.h>
#include <numeric>
#include <cuda_runtime.h>

//copy from https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer

namespace fast_transformers {
namespace layers {
namespace kernels {

static __global__
void add_bias_input_layernorm(float* out, const float* input, const float* bias, const float* gamma, const float* beta, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  for(int i = tid; i < n; i += blockDim.x)
    local_out += (float)(out[blockIdx.x * n + i] + input[blockIdx.x * n + i] + __ldg(&bias[i]));

  mean = blockReduceSum(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum((local_out - s_mean) * (local_out - s_mean));
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-6f;
  __syncthreads();

  for(int i = tid; i < n; i += blockDim.x)
    out[blockIdx.x * n + i] =
	    (float)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
}

template<>
void GPUAddBiasLayerNorm(float* out, const float* input, const float* bias,
  const float* gamma, const float* beta, int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n);
  if (n > 1024) {
    throw std::runtime_error("GPUAddBiasLayerNorm thread block size large than 1024");
  }
  add_bias_input_layernorm<<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
