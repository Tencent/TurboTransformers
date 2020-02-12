#pragma once
#include <cuda_runtime.h>

namespace fast_transformers {
namespace layers {
namespace kernels {

#define FINAL_MASK 0xffffffff

__inline__ __device__ float warpReduceSum(float val) {
  val += __shfl_xor_sync(FINAL_MASK, val, 16, 32);
  val += __shfl_xor_sync(FINAL_MASK, val, 8, 32);
  val += __shfl_xor_sync(FINAL_MASK, val, 4, 32);
  val += __shfl_xor_sync(FINAL_MASK, val, 2, 32);
  val += __shfl_xor_sync(FINAL_MASK, val, 1, 32);
  return val;
}

/*
 * Unorll for loop for warpreduce to
 * imporve instruction issue efficiency
 * ElemX means there are X numbers to be summed
 */

__inline__ __device__ void warpReduceSum_Elem2(float* val1, float* val2) {
  *val1 += __shfl_xor_sync(FINAL_MASK, *val1, 16, 32);
  *val2 += __shfl_xor_sync(FINAL_MASK, *val2, 16, 32);
  *val1 += __shfl_xor_sync(FINAL_MASK, *val1, 8, 32);
  *val2 += __shfl_xor_sync(FINAL_MASK, *val2, 8, 32);
  *val1 += __shfl_xor_sync(FINAL_MASK, *val1, 4, 32);
  *val2 += __shfl_xor_sync(FINAL_MASK, *val2, 4, 32);
  *val1 += __shfl_xor_sync(FINAL_MASK, *val1, 2, 32);
  *val2 += __shfl_xor_sync(FINAL_MASK, *val2, 2, 32);
  *val1 += __shfl_xor_sync(FINAL_MASK, *val1, 1, 32);
  *val2 += __shfl_xor_sync(FINAL_MASK, *val2, 1, 32);
}
__inline__ __device__ void warpReduceSum_Elem4(float* val0, float* val1,
                                               float* val2, float* val3) {
  *(val0) += __shfl_xor_sync(FINAL_MASK, *(val0), 16, 32);
  *(val1) += __shfl_xor_sync(FINAL_MASK, *(val1), 16, 32);
  *(val2) += __shfl_xor_sync(FINAL_MASK, *(val2), 16, 32);
  *(val3) += __shfl_xor_sync(FINAL_MASK, *(val3), 16, 32);

  *(val0) += __shfl_xor_sync(FINAL_MASK, *(val0), 8, 32);
  *(val1) += __shfl_xor_sync(FINAL_MASK, *(val1), 8, 32);
  *(val2) += __shfl_xor_sync(FINAL_MASK, *(val2), 8, 32);
  *(val3) += __shfl_xor_sync(FINAL_MASK, *(val3), 8, 32);

  *(val0) += __shfl_xor_sync(FINAL_MASK, *(val0), 4, 32);
  *(val1) += __shfl_xor_sync(FINAL_MASK, *(val1), 4, 32);
  *(val2) += __shfl_xor_sync(FINAL_MASK, *(val2), 4, 32);
  *(val3) += __shfl_xor_sync(FINAL_MASK, *(val3), 4, 32);

  *(val0) += __shfl_xor_sync(FINAL_MASK, *(val0), 2, 32);
  *(val1) += __shfl_xor_sync(FINAL_MASK, *(val1), 2, 32);
  *(val2) += __shfl_xor_sync(FINAL_MASK, *(val2), 2, 32);
  *(val3) += __shfl_xor_sync(FINAL_MASK, *(val3), 2, 32);

  *(val0) += __shfl_xor_sync(FINAL_MASK, *(val0), 1, 32);
  *(val1) += __shfl_xor_sync(FINAL_MASK, *(val1), 1, 32);
  *(val2) += __shfl_xor_sync(FINAL_MASK, *(val2), 1, 32);
  *(val3) += __shfl_xor_sync(FINAL_MASK, *(val3), 1, 32);
}

__inline__ __device__ void blockReduceSum_Elem4(float* val_list) {
  static __shared__ float shared[4][32];
  int lane_id = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSum_Elem4(val_list, val_list + 1, val_list + 2, val_list + 3);

  if (lane_id == 0) {
    shared[0][wid] = *(val_list);
    shared[1][wid] = *(val_list + 1);
    shared[2][wid] = *(val_list + 2);
    shared[3][wid] = *(val_list + 3);
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x >> 5)) {
    *(val_list + 0) = shared[0][lane_id];
    *(val_list + 1) = shared[1][lane_id];
    *(val_list + 2) = shared[2][lane_id];
    *(val_list + 3) = shared[3][lane_id];
  } else {
    *(val_list + 0) = 0.f;
    *(val_list + 1) = 0.f;
    *(val_list + 2) = 0.f;
    *(val_list + 3) = 0.f;
  }
  warpReduceSum_Elem4(val_list, val_list + 1, val_list + 2, val_list + 3);
}

__inline__ __device__ void blockReduceSum_Elem5(float* val_list, int size) {
  static __shared__ float shared[5][32];
  int lane_id = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSum_Elem5(val_list);

  if (lane_id == 0) {
    for (int i = 0; i < size; ++i) {
      shared[i][wid] = *(val_list + i);
    }
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x >> 5)) {
    *(val_list + 0) = shared[0][lane_id];
    *(val_list + 1) = shared[1][lane_id];
    *(val_list + 2) = shared[2][lane_id];
    *(val_list + 3) = shared[3][lane_id];
    *(val_list + 4) = shared[4][lane_id];
  } else {
    *(val_list + 0) = 0;
    *(val_list + 1) = 0;
    *(val_list + 2) = 0;
    *(val_list + 3) = 0;
    *(val_list + 4) = 0;
  }
  warpReduceSum_Elem5(val_list);
}

/* Calculate the sum of all elements in a block */
__inline__ __device__ void blockReduceSum_Elem2(float* val1, float* val2) {
  static __shared__ float shared1[32];
  static __shared__ float shared2[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSum_Elem2(val1, val2);

  if (lane == 0) {
    shared1[wid] = *val1;
    shared2[wid] = *val2;
  }

  __syncthreads();

  *val1 = (threadIdx.x < (blockDim.x >> 5)) ? shared1[lane] : (float)(0.0f);
  *val2 = (threadIdx.x < (blockDim.x >> 5)) ? shared2[lane] : (float)(0.0f);
  warpReduceSum_Elem2(val1, val2);
}

/* Calculate the sum of all elements in a block */
__inline__ __device__ float blockReduceSum(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (float)(0.0f);
  val = warpReduceSum(val);

  return val;
}

__inline__ __device__ float warpReduceMax(float val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
__inline__ __device__ float blockReduceMax(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x & 0x1f;  // in-warp idx
  int wid = threadIdx.x >> 5;     // warp idx

  val = warpReduceMax(val);  // get maxx in each warp

  if (lane == 0)  // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

#undef FINAL_MASK

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
