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
#include <array>
#include <stdexcept>
#include <string>
#ifdef FT_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include "absl/debugging/stacktrace.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
namespace easy_transformers {
namespace core {
namespace details {
/**
 * Implement enforce macros.
 *
 * Will throw `EnforceNotMet` when enforce check failed. Unlike GLOG and
 * std::assert, it:
 *   1. let the program catch the exception and recover from bad state.
 *   2. enforce carry a rich information about call stack. It is useful for
 * debugging.
 */

static constexpr size_t kStackLimit = 20UL;
static constexpr size_t kStackSkipCount = 1UL;
class EnforceNotMet : public std::exception {
 public:
  explicit EnforceNotMet(std::string msg) : msg_(std::move(msg)) {
    n_ = absl::GetStackTrace(stacks_.data(), stacks_.size(), kStackSkipCount);
  }

  const char *what() const noexcept override;

 private:
  mutable std::string msg_;
  std::array<void *, kStackLimit> stacks_{};
  size_t n_;
  mutable bool stack_added_{false};
};

#ifdef FT_WITH_CUDA
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
#endif
}  // namespace details

#if !defined(_WIN32)
#define FT_UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#else
#define FT_UNLIKELY(condition) (condition)
#endif

#define FT_THROW(...)                                        \
  do {                                                       \
    throw ::easy_transformers::core::details::EnforceNotMet( \
        absl::StrFormat(__VA_ARGS__));                       \
  } while (false)

#define FT_ENFORCE(cond, ...)                                              \
  do {                                                                     \
    if (FT_UNLIKELY(!(cond))) {                                            \
      throw ::easy_transformers::core::details::EnforceNotMet(             \
          absl::StrCat("enforce error", #cond,                             \
                       absl::StrFormat(" at %s:%d\n", __FILE__, __LINE__), \
                       absl::StrFormat(__VA_ARGS__)));                     \
    }                                                                      \
  } while (false)

#define FT_ENFORCE_EQ(a, b, ...) FT_ENFORCE((a) == (b), __VA_ARGS__)
#define FT_ENFORCE_NE(a, b, ...) FT_ENFORCE((a) != (b), __VA_ARGS__)
#define FT_ENFORCE_LT(a, b, ...) FT_ENFORCE((a) < (b), __VA_ARGS__)
#define FT_ENFORCE_LE(a, b, ...) FT_ENFORCE((a) <= (b), __VA_ARGS__)
#define FT_ENFORCE_GT(a, b, ...) FT_ENFORCE((a) > (b), __VA_ARGS__)
#define FT_ENFORCE_GE(a, b, ...) FT_ENFORCE((a) >= (b), __VA_ARGS__)

#ifdef FT_WITH_CUDA
#define FT_ENFORCE_CUDA_SUCCESS(COND, ...)                                   \
  do {                                                                       \
    auto __cond__ = (COND);                                                  \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);                         \
    constexpr auto __success_type__ =                                        \
        ::easy_transformers::core::details::CUDAStatusTrait<                 \
            __CUDA_STATUS_TYPE__>::kSuccess;                                 \
    if (FT_UNLIKELY(__cond__ != __success_type__)) {                         \
      std::string error_msg =                                                \
          std::string("[FT_ERROR] CUDA runtime error: ") +                   \
          ::easy_transformers::core::details::CUDAGetErrorString(__cond__) + \
          " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n";           \
      FT_THROW(error_msg.c_str());                                           \
    }                                                                        \
  } while (0)

#undef DEFINE_CUDA_STATUS_TYPE
#endif
}  // namespace core
}  // namespace easy_transformers
