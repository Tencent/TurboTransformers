/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#include <cuda_runtime.h>

namespace fast_transformers {
namespace layers {
namespace kernels {

#define FINAL_MASK 0xffffffff

// reduce two values to reduce syncthreads overhead
__inline__ __device__ void warpReduceSumTwoElemInline(float* val1,
                                                      float* val2) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    *val1 += __shfl_xor_sync(FINAL_MASK, *val1, mask, 32);
    *val2 += __shfl_xor_sync(FINAL_MASK, *val2, mask, 32);
  }
}

/* Calculate the sum of all elements in a block */
__inline__ __device__ void blockReduceSumTwoElemInline(float* val1,
                                                       float* val2) {
  static __shared__ float shared1[32];
  static __shared__ float shared2[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumTwoElemInline(val1, val2);

  if (lane == 0) {
    shared1[wid] = *val1;
    shared2[wid] = *val2;
  }

  __syncthreads();

  *val1 = (threadIdx.x < (blockDim.x >> 5)) ? shared1[lane] : (float)(0.0f);
  *val2 = (threadIdx.x < (blockDim.x >> 5)) ? shared2[lane] : (float)(0.0f);
  warpReduceSumTwoElemInline(val1, val2);
}

// inline reduce two values to remove syncthreads overhead
__inline__ __device__ void warpReduceSumv2(float* val1, float* val2) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    *val1 += __shfl_xor_sync(FINAL_MASK, *val1, mask, 32);
    *val2 += __shfl_xor_sync(FINAL_MASK, *val2, mask, 32);
  }
}

__inline__ __device__ void blockReduceSumv2(float* val1, float* val2) {
  static __shared__ float shared1[32];
  static __shared__ float shared2[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumv2(val1, val2);

  if (lane == 0) {
    shared1[wid] = *val1;
    shared2[wid] = *val2;
  }

  __syncthreads();

  *val1 = (threadIdx.x < (blockDim.x >> 5)) ? shared1[lane] : (float)(0.0f);
  *val2 = (threadIdx.x < (blockDim.x >> 5)) ? shared2[lane] : (float)(0.0f);
  warpReduceSumv2(val1, val2);
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
