

#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory.h>

#include <map>

#include "macros.h"

namespace turbo_transformers {
namespace core {

class CUDADeviceContext {
 public:
  CUDADeviceContext();

  ~CUDADeviceContext();

  static CUDADeviceContext& GetInstance() {
    static CUDADeviceContext instance;
    return instance;
  }

  void Wait() const;

  cudaStream_t stream() const;

  cublasHandle_t cublas_handle() const;

  int compute_major() const;

 private:
  cudaStream_t stream_;
  cublasHandle_t handle_;
  cudaDeviceProp device_prop_;
  DISABLE_COPY_AND_ASSIGN(CUDADeviceContext);
};

}  // namespace core
}  // namespace turbo_transformers
