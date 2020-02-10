#include <cuda_runtime.h>
#include <immintrin.h>
#include <numeric>
#include "fast_transformers/layers/kernels/gpu_common.h"
#include "fast_transformers/layers/kernels/gpu_layer_norm_kernel.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

//(m, n) block on n with blk_size
// TODO(jiaruifang) blk_size must be m times
template <bool isAdd>
__global__ void layer_norm_kernel_batch(float* out, const float* input,
                                        const float* bias, const float* gamma,
                                        const float* beta, int m, int n,
                                        int blk_size) {
  int tid = threadIdx.x;
  int offset = tid + blockIdx.x * blk_size * n;

#define MAX_BLK_INNNER_SIZE 2
  static __shared__ float s_mean[MAX_BLK_INNNER_SIZE];
  static __shared__ float s_variance[MAX_BLK_INNNER_SIZE];
  float local_out_buf[MAX_BLK_INNNER_SIZE];
  float sum_list[MAX_BLK_INNNER_SIZE * 2];
  float* sum_times_list = sum_list + MAX_BLK_INNNER_SIZE;
  const int max_blk_inner_size = MAX_BLK_INNNER_SIZE;

  float bias_local = 0.0;
  if (isAdd) {
    bias_local = __ldg(&bias[tid]);
  }
  float gamma_local = __ldg(&gamma[tid]);
  float beta_local = __ldg(&beta[tid]);
  int i;
  for (i = 0; i < blk_size; i += max_blk_inner_size) {
    int blk_size_inner = max_blk_inner_size;
#ifdef USE_UNROLL
#define set_sum_list_add(x)                                                 \
  local_out_buf[x] =                                                        \
      out[offset + (i + x) * n] + input[offset + (i + x) * n] + bias_local; \
  sum_list[x] = local_out_buf[x];                                           \
  sum_times_list[x] = local_out_buf[x] * local_out_buf[x];

#define set_sum_list(x)                           \
  local_out_buf[x] = input[offset + (i + x) * n]; \
  sum_list[x] = local_out_buf[x];                 \
  sum_times_list[x] = local_out_buf[x] * local_out_buf[x];

    if (isAdd) {
      set_sum_list_add(0);
      set_sum_list_add(1);
    } else {
      set_sum_list(0);
      set_sum_list(1);
    }
#else
    if (tid < n) {
      if (isAdd) {
        for (int j = 0; j < blk_size_inner; ++j) {
          local_out_buf[j] = out[offset + (i + j) * n] +
                             input[offset + (i + j) * n] + bias_local;
          sum_list[j] = local_out_buf[j];
          sum_times_list[j] = local_out_buf[j] * local_out_buf[j];
        }
      } else {
        for (int j = 0; j < blk_size_inner; ++j) {
          local_out_buf[j] = input[offset + (i + j) * n];
          sum_list[j] = local_out_buf[j];
          sum_times_list[j] = local_out_buf[j] * local_out_buf[j];
        }
      }
    } else {
      for (int j = 0; j < blk_size_inner; ++j) {
        local_out_buf[j] = 0.0;
      }
    }
#endif

#ifdef USE_UNROLL
    blockReduceSum_Elem4(sum_list);
#else
    for (int j = 0; j < blk_size_inner; ++j) {
      blockReduceSum_Elem2(sum_list + j, sum_times_list + j);
      __syncthreads();
    }
#endif

    if (tid == 0) {
#ifdef USE_UNROLL
      float mean;
#define set_shared_mean_variance(x) \
  mean = sum_list[x] / n;           \
  s_mean[x] = mean;                 \
  s_variance[x] = rsqrtf(sum_times_list[x] / n - mean * mean + 1e-6f)

      set_shared_mean_variance(0);
      set_shared_mean_variance(1);
#undef set_shared_mean_variance
#else
      for (int j = 0; j < blk_size_inner; ++j) {
        float mean = sum_list[j] / n;
        s_mean[j] = mean;
        s_variance[j] = rsqrtf(sum_times_list[j] / n - mean * mean + 1e-6f);
      }
#endif
    }
    __syncthreads();

    if (tid < n) {
#ifdef USE_UNROLL
      out[offset + (i)*n] =
          (local_out_buf[0] - s_mean[0]) * (gamma_local * s_variance[0]) +
          beta_local;
      out[offset + (i + 1) * n] =
          (local_out_buf[1] - s_mean[1]) * (gamma_local * s_variance[1]) +
          beta_local;
#else
      for (int j = 0; j < blk_size_inner; ++j) {
        out[offset + (i + j) * n] =
            (local_out_buf[j] - s_mean[j]) * s_variance[j] + beta_local;
      }
#endif
    }  // endif
  }    // for i
  // TODO(jiaruifang) dealing with th reminder
}

inline __device__ void get_mean_variance(float val, float* s_mean,
                                         float* s_variance, int n, int tid) {
  float sum1 = val, sum2 = val * val;
  blockReduceSum_Elem2(&sum1, &sum2);
  float mean = sum1 / n;
  float mean_2 = sum2 / n;

  if (tid == 0) {
    *s_mean = mean;
    *s_variance = rsqrtf(mean_2 - mean * mean + 1e-6f);
  }
  __syncthreads();
}

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
    local_out = (out[offset]);
  }

  get_mean_variance(local_out, &s_mean, &s_variance, n, tid);
  out[offset] = (local_out - s_mean) * s_variance * __ldg(&gamma[tid]) +
                __ldg(&beta[tid]);
}

template <bool isAdd>
static __global__ void layer_norm_kernel_nvidia(float* out, const float* input,
                                                const float* bias,
                                                const float* gamma,
                                                const float* beta, int m,
                                                int n) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  if (isAdd) {
    for (int i = tid; i < n; i += blockDim.x)
      local_out += (float)(out[blockIdx.x * n + i] + input[blockIdx.x * n + i] +
                           __ldg(&bias[i]));
  } else {
    for (int i = tid; i < n; i += blockDim.x)
      local_out = input[blockIdx.x * n + i];
  }

  mean = blockReduceSum(local_out);
  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum((local_out - s_mean) * (local_out - s_mean));
  if (threadIdx.x == 0) s_variance = variance / n + 1e-6f;
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x)
    out[blockIdx.x * n + i] =
        (float)(((local_out - s_mean) * rsqrtf(s_variance)) *
                    (float)(__ldg(&gamma[i])) +
                (float)(__ldg(&beta[i])));
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
