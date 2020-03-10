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

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "fast_transformers/core/enforce.h"
namespace fast_transformers {
namespace core {
namespace details {
static const std::string CUDAGetErrorString(cudaError_t error) {
  return cudaGetErrorString(error);
}

#define GetErrorNumCaseImpl(ENUM_VALUE) \
  do {                                  \
    if (error == ENUM_VALUE) {          \
      return #ENUM_VALUE;               \
    }                                   \
  } while (0)

static const std::string CUDAGetErrorString(cublasStatus_t error) {
  GetErrorNumCaseImpl(CUBLAS_STATUS_NOT_INITIALIZED);
  GetErrorNumCaseImpl(CUBLAS_STATUS_ALLOC_FAILED);
  GetErrorNumCaseImpl(CUBLAS_STATUS_INVALID_VALUE);
  GetErrorNumCaseImpl(CUBLAS_STATUS_ARCH_MISMATCH);
  GetErrorNumCaseImpl(CUBLAS_STATUS_MAPPING_ERROR);
  GetErrorNumCaseImpl(CUBLAS_STATUS_EXECUTION_FAILED);
  GetErrorNumCaseImpl(CUBLAS_STATUS_INTERNAL_ERROR);
  GetErrorNumCaseImpl(CUBLAS_STATUS_NOT_SUPPORTED);
  GetErrorNumCaseImpl(CUBLAS_STATUS_LICENSE_ERROR);
  return "<unknown>";
}

#undef GetErrorNumCaseImpl

template <typename T>
struct CUDAStatusTrait {};

#define DEFINE_CUDA_STATUS_TYPE(type, success_value) \
  template <>                                        \
  struct CUDAStatusTrait<type> {                     \
    using Type = type;                               \
    static constexpr Type kSuccess = success_value;  \
  }

DEFINE_CUDA_STATUS_TYPE(cudaError_t, cudaSuccess);
DEFINE_CUDA_STATUS_TYPE(cublasStatus_t, CUBLAS_STATUS_SUCCESS);
}  // namespace details

#define FT_ENFORCE_CUDA_SUCCESS(COND, ...)                                     \
  do {                                                                         \
    auto __cond__ = (COND);                                                    \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);                           \
    constexpr auto __success_type__ =                                          \
        details::CUDAStatusTrait<__CUDA_STATUS_TYPE__>::kSuccess;              \
    if (FT_UNLIKELY(__cond__ != __success_type__)) {                           \
      std::string error_msg = std::string("[FT_ERROR] CUDA runtime error: ") + \
                              details::CUDAGetErrorString(__cond__) + " " +    \
                              __FILE__ + ":" + std::to_string(__LINE__) +      \
                              " \n";                                           \
      FT_THROW(error_msg.c_str());                                             \
    }                                                                          \
  } while (0)

#undef DEFINE_CUDA_STATUS_TYPE

}  // namespace core
}  // namespace fast_transformers
