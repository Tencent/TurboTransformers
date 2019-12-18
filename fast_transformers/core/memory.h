#pragma once
#include "fast_transformers/core/enforce.h"
namespace fast_transformers {
namespace core {

extern void *align_alloc(size_t sz, size_t align = 64);

extern void *cuda_alloc(size_t sz);

extern void *cuda_free(void *data);

enum class MemcpyFlag { kCPU2GPU, kGPU2CPU, kCPU2CPU };

extern void FT_Memcpy(void *dst_data, const void *src_data, size_t data_size,
                      MemcpyFlag flag);

template <typename T>
inline T *align_alloc_t(size_t sz, size_t align = 64) {
  return reinterpret_cast<T *>(align_alloc(sz * sizeof(T), align));
}

}  // namespace core
}  // namespace fast_transformers
