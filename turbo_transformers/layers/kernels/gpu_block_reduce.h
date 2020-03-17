// Copyright 2020 Tencent
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cuda_runtime.h>
namespace turbo_transformers {
namespace layers {
namespace kernels {

#define FINAL_MASK 0xffffffff

static __inline__ __device__ float warpReduceSum(float val) {
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

static __inline__ __device__ void warpReduceSum_Elem2(float* val0,
                                                      float* val1) {
  float val0_tmp, val1_tmp;
#define WarpReduceSumOneStep(a, b)                       \
  val0_tmp = __shfl_xor_sync(FINAL_MASK, *(val0), a, b); \
  val1_tmp = __shfl_xor_sync(FINAL_MASK, *(val1), a, b); \
  *(val0) += val0_tmp;                                   \
  *(val1) += val1_tmp

  WarpReduceSumOneStep(16, 32);
  WarpReduceSumOneStep(8, 32);
  WarpReduceSumOneStep(4, 32);
  WarpReduceSumOneStep(2, 32);
  WarpReduceSumOneStep(1, 32);

#undef WarpReduceSumOneStep
}
static __inline__ __device__ void warpReduceSum_Elem4(float* val0, float* val1,
                                                      float* val2,
                                                      float* val3) {
  float val0_tmp, val1_tmp, val2_tmp, val3_tmp;
#define WarpReduceSumOneStep(a, b)                       \
  val0_tmp = __shfl_xor_sync(FINAL_MASK, *(val0), a, b); \
  val1_tmp = __shfl_xor_sync(FINAL_MASK, *(val1), a, b); \
  val2_tmp = __shfl_xor_sync(FINAL_MASK, *(val2), a, b); \
  val3_tmp = __shfl_xor_sync(FINAL_MASK, *(val3), a, b); \
  *(val0) += val0_tmp;                                   \
  *(val1) += val1_tmp;                                   \
  *(val2) += val2_tmp;                                   \
  *(val3) += val3_tmp

  WarpReduceSumOneStep(16, 32);
  WarpReduceSumOneStep(8, 32);
  WarpReduceSumOneStep(4, 32);
  WarpReduceSumOneStep(2, 32);
  WarpReduceSumOneStep(1, 32);
#undef WarpReduceSumOneStep
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

/* Calculate the sum of all elements in a block */
static __inline__ __device__ void blockReduceSum_Elem2(float* val1,
                                                       float* val2) {
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
static __inline__ __device__ float blockReduceSum(float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (float)(0.0f);
  val = warpReduceSum(val);

  return val;
}

// static
static __inline__ __device__ float warpReduceMax(float val) {
  val = max(val, __shfl_xor_sync(FINAL_MASK, val, 16, 32));
  val = max(val, __shfl_xor_sync(FINAL_MASK, val, 8, 32));
  val = max(val, __shfl_xor_sync(FINAL_MASK, val, 4, 32));
  val = max(val, __shfl_xor_sync(FINAL_MASK, val, 2, 32));
  val = max(val, __shfl_xor_sync(FINAL_MASK, val, 1, 32));
  return val;
}

static __inline__ __device__ void warpReduceMax_Elem2(float* val0,
                                                      float* val1) {
  float val0_tmp, val1_tmp;
#define WarpReduceMaxOneStep(a, b)                       \
  val0_tmp = __shfl_xor_sync(FINAL_MASK, *(val0), a, b); \
  val1_tmp = __shfl_xor_sync(FINAL_MASK, *(val1), a, b); \
  *(val0) = max(val0_tmp, *(val0));                      \
  *(val1) = max(val1_tmp, *(val1));

  WarpReduceMaxOneStep(16, 32);
  WarpReduceMaxOneStep(8, 32);
  WarpReduceMaxOneStep(4, 32);
  WarpReduceMaxOneStep(2, 32);
  WarpReduceMaxOneStep(1, 32);
#undef WarpReduceMaxOneStep
}

static __inline__ __device__ void warpReduceMax_Elem4(float* val0, float* val1,
                                                      float* val2,
                                                      float* val3) {
  float val0_tmp, val1_tmp, val2_tmp, val3_tmp;
#define WarpReduceMaxOneStep(a, b)                       \
  val0_tmp = __shfl_xor_sync(FINAL_MASK, *(val0), a, b); \
  val1_tmp = __shfl_xor_sync(FINAL_MASK, *(val1), a, b); \
  val2_tmp = __shfl_xor_sync(FINAL_MASK, *(val2), a, b); \
  val3_tmp = __shfl_xor_sync(FINAL_MASK, *(val3), a, b); \
  *(val0) = max(val0_tmp, *(val0));                      \
  *(val1) = max(val1_tmp, *(val1));                      \
  *(val2) = max(val2_tmp, *(val2));                      \
  *(val3) = max(val3_tmp, *(val3))

  WarpReduceMaxOneStep(16, 32);
  WarpReduceMaxOneStep(8, 32);
  WarpReduceMaxOneStep(4, 32);
  WarpReduceMaxOneStep(2, 32);
  WarpReduceMaxOneStep(1, 32);
#undef WarpReduceMaxOneStep
}

/* Calculate the maximum of all elements in a block */
// static
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

__inline__ __device__ void blockReduceMax_Elem2(float* val0, float* val1) {
  static __shared__ float shared[2][32];
  int lane_id = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceMax_Elem2(val0, val1);

  if (lane_id == 0) {
    shared[0][wid] = *(val0);
    shared[1][wid] = *(val1);
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x >> 5)) {
    *(val0) = shared[0][lane_id];
    *(val1) = shared[1][lane_id];
  } else {
    *(val0) = 0.f;
    *(val1) = 0.f;
  }
  warpReduceMax_Elem2(val0, val1);
}

__inline__ __device__ void blockReduceMax_Elem4(float* val_list) {
  static __shared__ float shared[4][32];
  int lane_id = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceMax_Elem4(val_list, val_list + 1, val_list + 2, val_list + 3);

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
  warpReduceMax_Elem4(val_list, val_list + 1, val_list + 2, val_list + 3);
}

#undef FINAL_MASK

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
