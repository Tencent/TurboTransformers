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
#if defined(__CUDACC__)  // && CUDA_VERSION >= 7050
#define CUDA_HALF
#include <cuda_fp16.h>
#endif
#include <fp16.h>

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
#if defined(TT_WITH_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    half tmp = __float2half(other);
    x = *reinterpret_cast<uint16_t *>(&tmp);
#else
    x = fp16_ieee_from_fp32_value(other);
#endif
  }
  template <class T>
  Half(const T &other) : Half(static_cast<float>(other)) {}

  operator float() const {
#if defined(TT_WITH_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    return __half2float(x);
#else
    return fp16_ieee_to_fp32_value(x);
#endif
  }
};
}  // namespace core
}  // namespace turbo_transformers
