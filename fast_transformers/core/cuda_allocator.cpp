#include "fast_transformers/core/cuda_allocator.h"

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

CUDAAllocator::~CUDAAllocator() { /*FreeCache(-1UL);*/
}

void CUDAAllocator::FreeCache(size_t size) {
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

void* CUDAAllocator::allocate(size_t size) {
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

size_t CUDAAllocator::allocation_size() const { return allocation_size_; }

void CUDAAllocator::free(void* memory, size_t size) {
  allocations_.emplace(size, memory);
  allocation_size_ += size;
}

}  // namespace core
}  // namespace fast_transformers
