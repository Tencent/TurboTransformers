

#include "allocator_impl.h"

#include "turbo_transformers/core/memory.h"
namespace turbo_transformers {
namespace core {
namespace allocator {
#ifdef TT_WITH_CUDA
static void *cuda_alloc(size_t sz) {
  void *device_mem;
  if (cudaMalloc((void **)&(device_mem), sz) != cudaSuccess) {
    throw BadAlloc("cudaMalloc failed.");
  }
  return device_mem;
}

static void cuda_free(void *data) { TT_ENFORCE_CUDA_SUCCESS(cudaFree(data)); }
#endif

void *allocate_impl(size_t size, DLDeviceType dev) {
  if (kDLCPU == dev) {
    return align_alloc(size);
  } else if (kDLGPU == dev) {
#ifdef TT_WITH_CUDA
    auto addr = cuda_alloc(size);
    return addr;
#endif
  } else {
    TT_THROW("Not supported devtype id as %d", dev);
  }
  return nullptr;
}

void free_impl(void *memory_addr, DLDeviceType dev) {
  if (kDLCPU == dev) {
    free(memory_addr);
  } else if (kDLGPU == dev) {
#ifdef TT_WITH_CUDA
    cuda_free(memory_addr);
#endif
  } else {
    TT_THROW("Not supported devtype id as %d", dev);
  }
}
}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
