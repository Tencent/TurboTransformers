#pragma once
#include <memory.h>
#include <map>

#include "fast_transformers/core/cuda_error.h"
#include "macros.h"

namespace fast_transformers {
namespace core {

class CUDAAllocator {
 public:
  explicit CUDAAllocator() : allocation_size_(0) {}
  virtual ~CUDAAllocator();

  static CUDAAllocator& GetInstance() {
    static CUDAAllocator instance;
    return instance;
  }

  void* allocate(size_t size);
  void free(void* memory, size_t size);
  void FreeCache(size_t size);
  size_t allocation_size() const;

 private:
  std::multimap<size_t, void*> allocations_;
  size_t allocation_size_;

  DISABLE_COPY_AND_ASSIGN(CUDAAllocator);
};

}  // namespace core
}  // namespace fast_transformers
