#include <cuda_runtime.h>
#include <immintrin.h>

#include <cub/cub.cuh>
#include <numeric>

#include "fast_transformers/layers/kernels/gpu_block_reduce.h"
#include "fast_transformers/layers/kernels/gpu_layer_norm_kernel.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

namespace {
template <typename T>
struct DataPair {
  __device__ __forceinline__ DataPair() {}
  __device__ __forceinline__ DataPair(const T& first, const T& second)
      : first(first), second(second) {}

  T first;
  T second;
};

template <typename T>
struct DataPairAddFunc {
  __device__ __forceinline__ DataPair<T> operator()(const DataPair<T>& p1,
                                                    const DataPair<T>& p2) {
    return DataPair<T>(p1.first + p2.first, p1.second + p2.second);
  }
};

template <bool isAdd, int BlockDim, typename T>
__global__ void cub_layer_norm_kernel(T* out, const T* input, const T* bias,
                                      const T* gamma, const T* beta, int m,
                                      int n) {
  using CubBlockReduce = cub::BlockReduce<DataPair<float>, BlockDim>;
  __shared__ typename CubBlockReduce::TempStorage temp_storage;
  __shared__ T s_mean;
  __shared__ T s_variance;

  int tid = threadIdx.x;
  T val1 = 0.0f, val2 = 0.0f;
  if (tid < n) {
    T tmp = input[blockIdx.x * n + tid];
    if (isAdd) {
      tmp += out[blockIdx.x * n + tid] + __ldg(&bias[tid]);
    }
    val1 = tmp;
    val2 = tmp * tmp;
  }

  auto pair =
      CubBlockReduce(temp_storage)
          .Reduce(DataPair<float>(val1, val2), DataPairAddFunc<float>());

  if (tid == 0) {
    s_mean = pair.first / n;
    s_variance = rsqrtf(pair.second / n - s_mean * s_mean + 1e-6f);
  }
  __syncthreads();

  if (tid < n) {
    out[blockIdx.x * n + tid] =
        (val1 - s_mean) * s_variance * __ldg(&gamma[tid]) + __ldg(&beta[tid]);
  }
}
}  // namespace

template <bool AddBias>
static __global__ void layer_norm_kernel(float* out, const float* input,
                                         const float* bias, const float* gamma,
                                         const float* beta, int m, int n) {
  int tid = threadIdx.x;
  int offset = blockIdx.x * n + tid;
  __shared__ float s_mean;
  __shared__ float s_variance;

  float local_out = 0.0f;
  if (AddBias) {
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

/*
template <bool AddBias, typename T>
void GPULayerNorm(T *out, const T *input, const T *bias, const T *gamma,
                  const T *beta, int m, int n, cudaStream_t stream) {
  if (n > 1024) {
    throw std::runtime_error("GPULayerNorm thread block size large than 1024.");
  }
  dim3 grid(m);
  dim3 block(1 << static_cast<int>(std::ceil(std::log2f(n))));

#define LayerNormKernelCase(AddMode, BlockDim, ...) \
  case (BlockDim):                                  \
    cub_layer_norm_kernel<(AddMode), (BlockDim)>    \
        <<<grid, block, 0, stream>>>(__VA_ARGS__);  \
    break

  switch (block.x) {
    LayerNormKernelCase(AddBias, 1024, out, input, bias, gamma, beta, m, n);
    LayerNormKernelCase(AddBias, 512, out, input, bias, gamma, beta, m, n);
    LayerNormKernelCase(AddBias, 128, out, input, bias, gamma, beta, m, n);
    LayerNormKernelCase(AddBias, 64, out, input, bias, gamma, beta, m, n);
    LayerNormKernelCase(AddBias, 32, out, input, bias, gamma, beta, m, n);
    LayerNormKernelCase(AddBias, 16, out, input, bias, gamma, beta, m, n);
    LayerNormKernelCase(AddBias, 8, out, input, bias, gamma, beta, m, n);
    LayerNormKernelCase(AddBias, 4, out, input, bias, gamma, beta, m, n);
    LayerNormKernelCase(AddBias, 2, out, input, bias, gamma, beta, m, n);
    LayerNormKernelCase(AddBias, 1, out, input, bias, gamma, beta, m, n);
  }
#undef LayerNormKernelCase
}
*/

template <bool AddBias, typename T>
void GPULayerNorm(T* out, const T* input, const T* bias, const T* gamma,
                  const T* beta, int m, int n, cudaStream_t stream) {
  dim3 block(n);
  if (block.x > 1024) {
    throw std::runtime_error("GPULayerNorm thread block size large than 1024");
  }
  dim3 grid(m);
  layer_norm_kernel<AddBias>
      <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}

template void GPULayerNorm<true>(float* out, const float* input,
                                 const float* bias, const float* gamma,
                                 const float* beta, int m, int n,
                                 cudaStream_t stream);
template void GPULayerNorm<false>(float* out, const float* input,
                                  const float* bias, const float* gamma,
                                  const float* beta, int m, int n,
                                  cudaStream_t stream);
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
