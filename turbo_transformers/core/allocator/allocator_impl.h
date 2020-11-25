

#pragma once
#include "turbo_transformers/core/memory.h"
#ifdef TT_WITH_CUDA
#include <cuda_runtime.h>

#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/core/cuda_enforce.cuh"
#endif
namespace turbo_transformers {
namespace core {
namespace allocator {

struct BadAlloc : public std::exception {
  explicit BadAlloc(std::string err_msg) : err_str_(err_msg) {}

  const char *what() const noexcept override { return err_str_.c_str(); }

  std::string err_str_;
};
extern void *allocate_impl(size_t size, DLDeviceType dev);

extern void free_impl(void *memory_addr, DLDeviceType dev);

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
