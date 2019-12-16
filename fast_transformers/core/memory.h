#pragma once
#include "fast_transformers/core/enforce.h"
namespace fast_transformers {
namespace core {

extern void *align_alloc(size_t sz, size_t align = 64);

#ifdef WITH_CUDA
extern void *cuda_alloc(size_t sz);

extern void *cuda_free(void* data);

enum FT_MemcpyFlag{
  FT_CPU2GPU = 0,
  FT_GPU2CPU = 1,
  FT_CPU2CPU = 2
};

template<typename T>
void FT_Memcpy(T* dst_data, const T* src_data, int64_t length, FT_MemcpyFlag flag);
#endif

template <typename T>
inline T *align_alloc_t(size_t sz, size_t align = 64) {
  return reinterpret_cast<T *>(align_alloc(sz * sizeof(T), align));
}


}  // namespace core
}  // namespace fast_transformers
