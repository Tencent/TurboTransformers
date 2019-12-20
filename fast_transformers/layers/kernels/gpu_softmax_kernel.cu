#include "fast_transformers/layers/kernels/gpu_softmax_kernel.h"
#include <immintrin.h>
#include <numeric>
#include <cuda_runtime.h>

//copy from https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer

namespace fast_transformers {
namespace layers {
namespace kernels {

/**
 * Multi-head attetion open sourced
 */
#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);

  return val;
}

template <typename T>
  __inline__ __device__
T warpReduceMax(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}


template <typename T>
__global__
void softmax_kernel(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len,
  const T scaler)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;

    __shared__ float s_sum, s_max;

    for(int i = 0; i < seq_len; ++i)
    {
      float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
      float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x % seq_len + batch_id * seq_len
      ] : 0.0f;

      //mask_val = (1.0f - mask_val) * -10000.0f;

      float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val): -1e20f;

      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0)
      {
        s_sum = sum_val + 1e-6f;
      }
      __syncthreads();

      if(threadIdx.x < seq_len)
        qk_buf_[threadIdx.x + qk_offset] = (T)(qk / s_sum);

      qk_offset += seq_len;
    }
}


template <typename T>
__global__
void softmax_kernel_v2(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num,
  const int seq_len, const float scaler)
{
    int batch_id = blockIdx.x / head_num / seq_len;
    int seq_id = blockIdx.x % seq_len;
    int qk_offset = blockIdx.x * seq_len;
    //int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;
    int mask_offset = batch_id * seq_len;

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    //float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] :0.0f;
    float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] :0.0f;

    //mask_val = (1.0f - mask_val) * -10000.0f;

    float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val) : -1e20f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

void GPUSoftmaxMask(float* qk_buf, const float* attr_mask,
                        int64_t batch_size, int64_t head_num, int64_t seq_len,
                        float scale, cudaStream_t stream) {
  dim3 block, grid;
  if(seq_len <= 32)
    block.x = 32;
  else if(seq_len > 32 && seq_len <= 64)
    block.x = 64;
  else if(seq_len > 64 && seq_len <= 128)
    block.x = 128;
  else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
  else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
  else
    block.x = 1024;
  //assert(n > 1024);
  if(batch_size * head_num <= 120)
  {
    grid.x = batch_size * head_num * seq_len;
    softmax_kernel_v2<float><<<grid, block, 0, stream>>>(qk_buf, attr_mask, batch_size, head_num, seq_len, scale);
  }
  else
  {
    grid.x = batch_size * head_num;
    softmax_kernel<float><<<grid, block, 0, stream>>>(qk_buf, attr_mask, batch_size, head_num, seq_len, scale);
  }
}

#undef FINAL_MASK

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
