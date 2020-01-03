#include "fast_transformers/core/cuda_device_context.h"

#include "fast_transformers/core/enforce.h"
#include "fast_transformers/core/memory.h"

namespace fast_transformers {
namespace core {

CUDADeviceContext::CUDADeviceContext() {
  cudaStreamCreate(&stream_);
  cublas_handle_.reset(new CublasHandleHolder(stream_));
}

void CUDADeviceContext::Wait() const {
  cudaError_t e_sync = cudaSuccess;
  e_sync = cudaStreamSynchronize(stream_);
  FT_ENFORCE_CUDA_SUCCESS(e_sync);
}

cudaStream_t CUDADeviceContext::stream() const { return stream_; }

int CUDADeviceContext::device_count() const {
  int count = 0;
  FT_ENFORCE_CUDA_SUCCESS(cudaGetDeviceCount(&count));
  return count;
}

CUDADeviceContext::~CUDADeviceContext() {
  Wait();
  cublas_handle_.reset();
  FT_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(stream_));
}

}  // namespace core
}  // namespace fast_transformers
