#include "fast_transformers/layers/kernels/gpu_layer_norm_kernel.h"
#include <immintrin.h>
#include <numeric>
#include <cuda_runtime.h>

//copy from https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer

namespace fast_transformers {
namespace layers {
namespace kernels {

#define FINAL_MASK 0xffffffff

template <typename T>
static __inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
static __inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}


template <typename T>
static __global__
void add_bias_input_layernorm(T* out, const T* input, const T* bias, const T* gamma, const T* beta, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  for(int i = tid; i < n; i += blockDim.x)
    local_out += (float)(out[blockIdx.x * n + i] + input[blockIdx.x * n + i] + __ldg(&bias[i]));

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-6f;
  __syncthreads();

  for(int i = tid; i < n; i += blockDim.x)
    out[blockIdx.x * n + i] =
	    (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
}

template<typename T>
void GPUAddBiasLayerNorm(T* out, const T* input, const T* bias,
  const T* gamma, const T* beta, int m, int n, cudaStream_t stream)
{
  //assert(n < 1024);
  dim3 grid(m);
  dim3 block(n);
  add_bias_input_layernorm<T><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}

template void GPUAddBiasLayerNorm<float>(float* out, const float* input, const float* bias,
  const float* gamma, const float* beta, int m, int n, cudaStream_t stream);

#undef FINAL_MASK

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
