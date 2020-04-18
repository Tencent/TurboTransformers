// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

#pragma once
#include <cuda_runtime.h>
#include "turbo_transformers/layers/types.h"
#include <algorithm>

namespace turbo_transformers {
namespace layers {
namespace kernels {

using ReduceType = types::ReduceType;

#define FINAL_MASK 0xffffffff

template <typename Type, Type t, int NumElem>
__inline__ __device__ void blockReduce(float* val_list);

// use template to make code more concise
template <ReduceType TypeVal, int NumElem>
__inline__ __device__ void warpReduce(float* val_list);

// static
template <>
__inline__ __device__ void warpReduce<ReduceType::kMax, 1>(float* val_list) {
  *val_list = max(*val_list, __shfl_xor_sync(FINAL_MASK, *val_list, 16, 32));
  *val_list = max(*val_list, __shfl_xor_sync(FINAL_MASK, *val_list, 8, 32));
  *val_list = max(*val_list, __shfl_xor_sync(FINAL_MASK, *val_list, 4, 32));
  *val_list = max(*val_list, __shfl_xor_sync(FINAL_MASK, *val_list, 2, 32));
  *val_list = max(*val_list, __shfl_xor_sync(FINAL_MASK, *val_list, 1, 32));
}

template <>
__inline__ __device__ void warpReduce<ReduceType::kMax, 2>(float* val_list) {
  float val0_tmp, val1_tmp;
#define WarpReduceMaxOneStep(a, b)                               \
  val0_tmp = __shfl_xor_sync(FINAL_MASK, *(val_list), a, b);     \
  val1_tmp = __shfl_xor_sync(FINAL_MASK, *(val_list + 1), a, b); \
  *(val_list) = max(val0_tmp, *(val_list));                      \
  *(val_list + 1) = max(val1_tmp, *(val_list + 1));

  WarpReduceMaxOneStep(16, 32);
  WarpReduceMaxOneStep(8, 32);
  WarpReduceMaxOneStep(4, 32);
  WarpReduceMaxOneStep(2, 32);
  WarpReduceMaxOneStep(1, 32);
#undef WarpReduceMaxOneStep
}

template <>
__inline__ __device__ void warpReduce<ReduceType::kMax, 4>(float* val_list) {
  float val0_tmp, val1_tmp, val2_tmp, val3_tmp;
#define WarpReduceMaxOneStep(a, b)                               \
  val0_tmp = __shfl_xor_sync(FINAL_MASK, *(val_list + 0), a, b); \
  val1_tmp = __shfl_xor_sync(FINAL_MASK, *(val_list + 1), a, b); \
  val2_tmp = __shfl_xor_sync(FINAL_MASK, *(val_list + 2), a, b); \
  val3_tmp = __shfl_xor_sync(FINAL_MASK, *(val_list + 3), a, b); \
  *(val_list + 0) = max(val0_tmp, *(val_list + 0));              \
  *(val_list + 1) = max(val1_tmp, *(val_list + 1));              \
  *(val_list + 2) = max(val2_tmp, *(val_list + 2));              \
  *(val_list + 3) = max(val3_tmp, *(val_list + 3))

  WarpReduceMaxOneStep(16, 32);
  WarpReduceMaxOneStep(8, 32);
  WarpReduceMaxOneStep(4, 32);
  WarpReduceMaxOneStep(2, 32);
  WarpReduceMaxOneStep(1, 32);
#undef WarpReduceMaxOneStep
}

template <>
__inline__ __device__ void warpReduce<ReduceType::kSum, 1>(float* val_list) {
  *val_list += __shfl_xor_sync(FINAL_MASK, *val_list, 16, 32);
  *val_list += __shfl_xor_sync(FINAL_MASK, *val_list, 8, 32);
  *val_list += __shfl_xor_sync(FINAL_MASK, *val_list, 4, 32);
  *val_list += __shfl_xor_sync(FINAL_MASK, *val_list, 2, 32);
  *val_list += __shfl_xor_sync(FINAL_MASK, *val_list, 1, 32);
}

/*
 * Unorll for loop for warpreduce to
 * imporve instruction issue efficiency
 * ElemX means there are X numbers to be summed
 */

template <>
__inline__ __device__ void warpReduce<ReduceType::kSum, 2>(float* val_list) {
  float val0_tmp, val1_tmp;
#define WarpReduceSumOneStep(a, b)                               \
  val0_tmp = __shfl_xor_sync(FINAL_MASK, *(val_list + 0), a, b); \
  val1_tmp = __shfl_xor_sync(FINAL_MASK, *(val_list + 1), a, b); \
  *(val_list + 0) += val0_tmp;                                   \
  *(val_list + 1) += val1_tmp

  WarpReduceSumOneStep(16, 32);
  WarpReduceSumOneStep(8, 32);
  WarpReduceSumOneStep(4, 32);
  WarpReduceSumOneStep(2, 32);
  WarpReduceSumOneStep(1, 32);

#undef WarpReduceSumOneStep
}

template <>
__inline__ __device__ void warpReduce<ReduceType::kSum, 4>(float* val_list) {
  float val0_tmp, val1_tmp, val2_tmp, val3_tmp;
#define WarpReduceSumOneStep(a, b)                               \
  val0_tmp = __shfl_xor_sync(FINAL_MASK, *(val_list + 0), a, b); \
  val1_tmp = __shfl_xor_sync(FINAL_MASK, *(val_list + 1), a, b); \
  val2_tmp = __shfl_xor_sync(FINAL_MASK, *(val_list + 2), a, b); \
  val3_tmp = __shfl_xor_sync(FINAL_MASK, *(val_list + 3), a, b); \
  *(val_list + 0) += val0_tmp;                                   \
  *(val_list + 1) += val1_tmp;                                   \
  *(val_list + 2) += val2_tmp;                                   \
  *(val_list + 3) += val3_tmp

  WarpReduceSumOneStep(16, 32);
  WarpReduceSumOneStep(8, 32);
  WarpReduceSumOneStep(4, 32);
  WarpReduceSumOneStep(2, 32);
  WarpReduceSumOneStep(1, 32);
#undef WarpReduceSumOneStep
}

template <ReduceType TypeVal, int NumElem>
__inline__ __device__ void blockReduce(float* val_list) {
  const int n = NumElem;
  static __shared__ float shared[n][32];
  int lane_id = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduce<TypeVal, NumElem>(val_list);

  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < n; ++i) {
      shared[i][wid] = *(val_list + i);
    }
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x >> 5)) {
#pragma unroll
    for (int i = 0; i < n; ++i) {
      *(val_list + i) = shared[i][lane_id];
    }
  } else {
#pragma unroll
    for (int i = 0; i < n; ++i) {
      *(val_list + i) = 0.f;
    }
  }
  warpReduce<TypeVal, NumElem>(val_list);
}

#undef FINAL_MASK

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
