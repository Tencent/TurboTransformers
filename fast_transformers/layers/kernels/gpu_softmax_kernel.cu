#include <cuda_runtime.h>
#include <immintrin.h>

#include <cub/cub.cuh>
#include <numeric>

#include "fast_transformers/layers/kernels/gpu_block_reduce.h"
#include "fast_transformers/layers/kernels/gpu_softmax_kernel.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

namespace {
template <int BlockDim>
__global__ void cub_softmax_kernel(float* qk_buf_, const float* attr_mask,
                                   const int batch_size, const int head_num,
                                   const int seq_len, const float scaler) {
  using CubBlockReduce = cub::BlockReduce<float, BlockDim>;
  __shared__ typename CubBlockReduce::TempStorage temp_storage;
  __shared__ float s_sum, s_max;

  int batch_id = blockIdx.x / head_num / seq_len;
  int qk_offset = blockIdx.x * seq_len;
  int mask_offset = batch_id * seq_len;

  float qk = threadIdx.x < seq_len ? qk_buf_[threadIdx.x + qk_offset] : 0.0f;
  float mask_val =
      threadIdx.x < seq_len ? attr_mask[threadIdx.x + mask_offset] : 0.0f;
  // mask_val = (1.0f - mask_val) * -10000.0f;
  float tmp = threadIdx.x < seq_len ? (qk * scaler + mask_val) : -1e20f;

  float max_val = CubBlockReduce(temp_storage).Reduce(tmp, cub::Max());
  if (threadIdx.x == 0) s_max = max_val;
  __syncthreads();

  float qk_tmp = threadIdx.x < seq_len ? __expf((tmp - s_max)) : 0.0f;
  float sum_val = CubBlockReduce(temp_storage).Reduce(qk_tmp, cub::Sum());

  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-6f;
  }
  __syncthreads();

  if (threadIdx.x < seq_len)
    qk_buf_[threadIdx.x + qk_offset] = (qk_tmp / s_sum);
}
}  // namespace

// unroll for loop using unroll size as 4.
// blk_size can be arbitary positive intergers.
// For sake of the effecieny blk_size should better be 4x.
__global__ void softmax_kernel_unroll4(float* qk_buf_, const float* attr_mask,
                                       int batch_size, int head_num,
                                       int seq_len, float scaler,
                                       int blk_size) {
  int batch_id = blockIdx.x * blk_size / seq_len / head_num;
  int tid = threadIdx.x;
  int qk_offset = tid + blockIdx.x * seq_len * blk_size;

  const int loop_unroll_size = 4;
  static __shared__ float s_sum[loop_unroll_size];
  static __shared__ float s_max[loop_unroll_size];
  float qk_list[loop_unroll_size];
  float qk_sum_list[loop_unroll_size];
  float qk_max_list[loop_unroll_size];

  float mask_val = tid < seq_len ? attr_mask[tid + batch_id * seq_len] : 0.0f;
  float qk;
  int i;
  int iter_size = blk_size / loop_unroll_size * loop_unroll_size;
  for (i = 0; i < iter_size; i += loop_unroll_size) {
    int blk_size_inner = loop_unroll_size;
    if (threadIdx.x < seq_len) {
      int qk_buf_offset = qk_offset + i * seq_len;
      qk = qk_buf_[qk_buf_offset];
      qk_max_list[0] = qk_list[0] = ((qk * scaler + mask_val));

      qk = qk_buf_[qk_buf_offset + seq_len];
      qk_max_list[1] = qk_list[1] = ((qk * scaler + mask_val));

      qk = qk_buf_[qk_buf_offset + 2 * seq_len];
      qk_max_list[2] = qk_list[2] = ((qk * scaler + mask_val));

      qk = qk_buf_[qk_buf_offset + 3 * seq_len];
      qk_max_list[3] = qk_list[3] = ((qk * scaler + mask_val));
    } else {
      qk_max_list[0] = -1e20f;
      qk_max_list[1] = -1e20f;
      qk_max_list[2] = -1e20f;
      qk_max_list[3] = -1e20f;
    }

    // max-trick
    blockReduceMax_Elem4(qk_max_list);
    if (tid == 0) {
      s_max[0] = qk_max_list[0];
      s_max[1] = qk_max_list[1];
      s_max[2] = qk_max_list[2];
      s_max[3] = qk_max_list[3];
    }
    __syncthreads();

    if (tid < seq_len) {
      qk_sum_list[0] = qk_list[0] = __expf((qk_list[0] - s_max[0]));
      qk_sum_list[1] = qk_list[1] = __expf((qk_list[1] - s_max[1]));
      qk_sum_list[2] = qk_list[2] = __expf((qk_list[2] - s_max[2]));
      qk_sum_list[3] = qk_list[3] = __expf((qk_list[3] - s_max[3]));
    } else {
      qk_sum_list[0] = 0.0f;
      qk_sum_list[1] = 0.0f;
      qk_sum_list[2] = 0.0f;
      qk_sum_list[3] = 0.0f;
    }
    __syncthreads();

    blockReduceSum_Elem4(qk_sum_list);
    if (tid == 0) {
      s_sum[0] = qk_sum_list[0] + 1e-6f;
      s_sum[1] = qk_sum_list[1] + 1e-6f;
      s_sum[2] = qk_sum_list[2] + 1e-6f;
      s_sum[3] = qk_sum_list[3] + 1e-6f;
    }
    __syncthreads();

    if (threadIdx.x < seq_len) {
      qk_buf_[qk_offset + seq_len * (i + 0)] = (qk_list[0] / s_sum[0]);
      qk_buf_[qk_offset + seq_len * (i + 1)] = (qk_list[1] / s_sum[1]);
      qk_buf_[qk_offset + seq_len * (i + 2)] = (qk_list[2] / s_sum[2]);
      qk_buf_[qk_offset + seq_len * (i + 3)] = (qk_list[3] / s_sum[3]);
    }  // endif
  }    // for i
  // dealing with the remaining lines
  int blk_size_inner = blk_size % loop_unroll_size;
  if (blk_size_inner == 0) return;
  if (threadIdx.x < seq_len) {
    for (int j = 0; j < blk_size_inner; ++j) {
      qk = qk_buf_[qk_offset + (i + j) * seq_len];
      qk_max_list[j] = qk_list[j] = qk * scaler + mask_val;
    }
  } else {
    for (int j = 0; j < blk_size_inner; ++j) {
      qk_max_list[j] = qk_list[j] = -1e20f;
    }
  }
  // max-trick
  for (int j = 0; j < blk_size_inner; ++j) {
    qk_sum_list[j] = blockReduceMax(qk_max_list[j]);
  }
  if (tid == 0) {
    for (int j = 0; j < blk_size_inner; ++j) {
      s_max[j] = qk_max_list[j];
    }
  }
  __syncthreads();

  if (tid < seq_len) {
    for (int j = 0; j < blk_size_inner; ++j)
      qk_sum_list[0] = qk_list[0] = __expf((qk_list[0] - s_max[0]));
  } else {
    for (int j = 0; j < blk_size_inner; ++j) qk_sum_list[0] = 0.0f;
  }
  for (int j = 0; j < blk_size_inner; ++j) {
    qk_sum_list[j] = blockReduceSum(qk_list[j]);
  }

  if (tid == 0) {
    for (int j = 0; j < blk_size_inner; ++j) {
      s_sum[j] = qk_sum_list[j] + 1e-6f;
    }
  }
  __syncthreads();
  if (threadIdx.x < seq_len) {
    for (int j = 0; j < blk_size_inner; ++j) {
      qk_buf_[qk_offset + seq_len * (i + j)] = (qk_list[j] / s_sum[j]);
    }
  }  // endif
}

// nvidia version block size on the high dimension is always 1, may lead to
// low occupancy.
__global__ void softmax_kernel_noblk(float* qk_buf_, const float* attr_mask,
                                     const int batch_size, const int head_num,
                                     const int seq_len, const float scaler) {
  int batch_id = blockIdx.x / head_num / seq_len;
  int qk_offset = blockIdx.x * seq_len;
  int mask_offset = batch_id * seq_len;

  __shared__ float s_sum, s_max;

  float qk = threadIdx.x < seq_len ? qk_buf_[threadIdx.x + qk_offset] : 0.0f;
  float mask_val =
      threadIdx.x < seq_len ? attr_mask[threadIdx.x + mask_offset] : 0.0f;

  // mask_val = (1.0f - mask_val) * -10000.0f;
  float tmp = threadIdx.x < seq_len ? (qk * scaler + mask_val) : -1e20f;
  float max_val = blockReduceMax(tmp);
  if (threadIdx.x == 0) s_max = max_val;
  __syncthreads();

  float qk_tmp = threadIdx.x < seq_len ? __expf((tmp - s_max)) : 0.0f;
  float sum_val = blockReduceSum(qk_tmp);

  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-6f;
  }
  __syncthreads();

  if (threadIdx.x < seq_len)
    qk_buf_[threadIdx.x + qk_offset] = (qk_tmp / s_sum);
}

/*
template <>
void GPUSoftmaxMask(float* qk_buf, const float* attr_mask, int64_t batch_size,
                    int64_t head_num, int64_t seq_len, float scale,
                    cudaStream_t stream) {
  dim3 block, grid;
  int blk_size;
  int high_dim_size = batch_size * head_num * seq_len;

  // block size must be 32x, so warp reduce can work
  block.x = (seq_len + 31) / 32 * 32;
  blk_size = 4;
  // In the senario of BERT inference, high_dim_size is 4x because head_num is
  // 12, 40 is set according to experimental results on Tesla M40.
  if (high_dim_size < 40 * 12 || high_dim_size % blk_size != 0) {
    grid.x = high_dim_size;
    softmax_kernel_noblk<<<grid, block, 0, stream>>>(
        qk_buf, attr_mask, batch_size, head_num, seq_len, scale);
  } else {
    grid.x = high_dim_size / blk_size;
    softmax_kernel_unroll4<<<grid, block, 0, stream>>>(
        qk_buf, attr_mask, batch_size, head_num, seq_len, scale, blk_size);
  }
}
*/

#define SOFTMAX_KERNEL_CASE(BlockDim, ...)                                 \
  case (BlockDim):                                                         \
    cub_softmax_kernel<BlockDim><<<grid, block, 0, stream>>>(__VA_ARGS__); \
    break

#define RUN_KERNEL(...)                       \
  do {                                        \
    switch (block.x) {                        \
      SOFTMAX_KERNEL_CASE(1024, __VA_ARGS__); \
      SOFTMAX_KERNEL_CASE(512, __VA_ARGS__);  \
      SOFTMAX_KERNEL_CASE(128, __VA_ARGS__);  \
      SOFTMAX_KERNEL_CASE(64, __VA_ARGS__);   \
      SOFTMAX_KERNEL_CASE(32, __VA_ARGS__);   \
      SOFTMAX_KERNEL_CASE(16, __VA_ARGS__);   \
      SOFTMAX_KERNEL_CASE(8, __VA_ARGS__);    \
      SOFTMAX_KERNEL_CASE(4, __VA_ARGS__);    \
      SOFTMAX_KERNEL_CASE(2, __VA_ARGS__);    \
      SOFTMAX_KERNEL_CASE(1, __VA_ARGS__);    \
    }                                         \
  } while (0)

template <>
void GPUSoftmaxMask(float* qk_buf, const float* attr_mask, int64_t batch_size,
                    int64_t head_num, int64_t seq_len, float scale,
                    cudaStream_t stream) {
  dim3 block, grid;
  int blk_size = 1;
  int high_dim_size = batch_size * head_num * seq_len;

  // block size must be 32x, so warp reduce can work
  block.x = (seq_len + 31) / 32 * 32;
  grid.x = high_dim_size;
  RUN_KERNEL(qk_buf, attr_mask, batch_size, head_num, seq_len, scale);
}

#undef RUN_KERNEL
#undef SOFTMAX_KERNEL_CASE
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
