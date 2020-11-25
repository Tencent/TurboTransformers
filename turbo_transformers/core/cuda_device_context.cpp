

#include "turbo_transformers/core/cuda_device_context.h"

#include "turbo_transformers/core/cuda_enforce.cuh"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/core/memory.h"

namespace turbo_transformers {
namespace core {

CUDADeviceContext::CUDADeviceContext() {
  TT_ENFORCE_CUDA_SUCCESS(cudaStreamCreate(&stream_));
  TT_ENFORCE_CUDA_SUCCESS(cublasCreate(&handle_));
  TT_ENFORCE_CUDA_SUCCESS(cublasSetStream(handle_, stream_));
  TT_ENFORCE_CUDA_SUCCESS(cudaGetDeviceProperties(&device_prop_, 0));
}

void CUDADeviceContext::Wait() const {
  cudaError_t e_sync = cudaSuccess;
  e_sync = cudaStreamSynchronize(stream_);
  TT_ENFORCE_CUDA_SUCCESS(e_sync);
}

cudaStream_t CUDADeviceContext::stream() const { return stream_; }

cublasHandle_t CUDADeviceContext::cublas_handle() const { return handle_; }

int CUDADeviceContext::compute_major() const { return device_prop_.major; }

CUDADeviceContext::~CUDADeviceContext() {
  Wait();
  TT_ENFORCE_CUDA_SUCCESS(cublasDestroy(handle_));
  TT_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(stream_));
}

}  // namespace core
}  // namespace turbo_transformers
