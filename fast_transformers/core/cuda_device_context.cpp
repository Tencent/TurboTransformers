#include "fast_transformers/core/cuda_device_context.h"

#include "fast_transformers/core/enforce.h"
#include "fast_transformers/core/memory.h"

namespace fast_transformers {
namespace core {

struct BadAlloc : public std::exception {
  explicit BadAlloc(std::string err_msg, const char* file, int line)
      : err_str_(err_msg) {}

  const char* what() const noexcept override { return err_str_.c_str(); }

  std::string err_str_;
};

CUDADeviceContext::CUDADeviceContext() : allocation_size_(0) {
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
  FreeCache(-1UL);
}

void CUDADeviceContext::FreeCache(size_t size) {
  if (FT_UNLIKELY(size == 0)) return;
  size_t cur = 0;
  while (!allocations_.empty()) {  // free the largest
    auto it = --allocations_.end();
    cur += it->first;
    cuda_free(it->second);
    allocation_size_ -= it->first;
    allocations_.erase(it);
    if (cur >= size) return;
  }
}

void* CUDADeviceContext::allocate(size_t size) {
  auto it = allocations_.lower_bound(size);
  if (it != allocations_.end() && it->first < size * 2) {
    void* result = it->second;
    allocation_size_ -= it->first;
    allocations_.erase(it);
    return result;
  }

  try {
    return cuda_alloc(size);
  } catch (BadAlloc&) {
    std::cerr << "I want to have " << size << " Byte data" << std::endl;
    FreeCache(size);
    return cuda_alloc(size);
  }
}

void CUDADeviceContext::free(void* memory, size_t size) {
  allocations_.emplace(size, memory);
  allocation_size_ += size;
}

}  // namespace core
}  // namespace fast_transformers
