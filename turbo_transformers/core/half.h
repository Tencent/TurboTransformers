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
#if defined(__CUDACC__)
#define CUDA_HALF
#include <cuda_fp16.h>
#endif
#include <fp16.h>

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace turbo_transformers {
namespace core {

struct alignas(2) Half {
  std::uint16_t x;

  Half() = default;

  Half(const Half &f) = default;

  Half &operator=(const Half &f) = default;

  Half &operator=(Half &&f) = default;

  ~Half() = default;

  Half(float other) {
#if defined(CUDA_HALF)
    half tmp = __float2half(other);
    x = *reinterpret_cast<uint16_t *>(&tmp);
#else
    x = fp16_ieee_from_fp32_value(other);
#endif
  }

  template <class T>
  Half(const T &other) : Half(static_cast<float>(other)) {}

  inline operator float() const {
#if defined(CUDA_HALF)
    return __half2float(x);
#else
    return fp16_ieee_to_fp32_value(x);
#endif
  }
};

#ifdef CUDA_HALF
DEVICE __half operator+(const __half &a, const __half &b) {
  return __hadd(a, b);
}
#endif

#if defined(CUDA_HALF) && __CUDA_ARCH__ >= 600
DEVICE __half2 operator+(const __half2 &a, const __half2 &b) {
  return __hadd2(a, b);
}
#endif
}  // namespace core
}  // namespace turbo_transformers
