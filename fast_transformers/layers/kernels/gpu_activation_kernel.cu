#include "fast_transformers/layers/kernels/gpu_activation_kernel.h"
#include <immintrin.h>
#include <numeric>
#include <cuda_runtime.h>

namespace fast_transformers {
namespace layers {
namespace kernels {
template <typename T>
static __inline__ __device__
T gelu(T x)
{
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <typename T>
static __global__
void add_bias_act(T* out, const T* bias, int batch_size, int feature_dim)
{
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = feature_dim / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m){
      val = out[tid + i * blockDim.x + row_id * n]+ reg_bias;
      out[tid + i * blockDim.x + row_id * n] = gelu<T>(val);
      row_id += gridDim.x;
    }
  }
}

template <typename T>
void GPUAddBiasGeLUActKernel(const T* bias_data, T* out_data, int64_t batch_size, int64_t feature_dim, cudaStream_t stream) {
  dim3 grid(batch_size / 4);
  dim3 block(feature_dim / 4);
  add_bias_act<T><<<grid, block, 0, stream>>>(out_data, bias_data, batch_size, feature_dim);
}


template void GPUAddBiasGeLUActKernel<float>(const float* bias_data, float* out_data, int64_t batch_size, int64_t feature_dim, cudaStream_t stream);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
