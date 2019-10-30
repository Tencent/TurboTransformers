#pragma once
#include "fast_transformers/core/enforce.h"
namespace fast_transformers {
namespace core {

extern void *align_alloc(size_t sz, size_t align = 64);

template <typename T> inline T *align_alloc_t(size_t sz, size_t align = 64) {
  return reinterpret_cast<T *>(align_alloc(sz * sizeof(T), align));
}

} // namespace core
} // namespace fast_transformers
