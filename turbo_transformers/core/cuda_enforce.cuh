#pragma once
#include <string>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace turbo_transformers {
namespace core {

namespace details {
static inline const std::string CUDAGetErrorString(cudaError_t error) {
  return cudaGetErrorString(error);
}

#define GetErrorNumCaseImpl(ENUM_VALUE) \
  do {                                  \
    if (error == ENUM_VALUE) {          \
      return #ENUM_VALUE;               \
    }                                   \
  } while (0)

static inline const std::string CUDAGetErrorString(cublasStatus_t error) {
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
}

#define TT_ENFORCE_CUDA_SUCCESS(COND, ...)                                    \
  do {                                                                        \
    auto __cond__ = (COND);                                                   \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);                          \
    constexpr auto __success_type__ =                                         \
        ::turbo_transformers::core::details::CUDAStatusTrait<                 \
            __CUDA_STATUS_TYPE__>::kSuccess;                                  \
    if (__cond__ != __success_type__) {                          \
      std::string error_msg =                                                 \
          std::string("[TT_ERROR] CUDA runtime error: ") +                    \
          ::turbo_transformers::core::details::CUDAGetErrorString(__cond__) + \
          " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n";            \
      printf("%s\n", error_msg.c_str());                                            \
      exit(0); \
    }                                                                         \
  } while (0)

#undef DEFINE_CUDA_STATUS_TYPE

}
}
