#include "fast_transformers/core/memory.h"

namespace fast_transformers {
namespace core {
void *align_alloc(size_t sz, size_t align) {
  void *aligned_mem;
  FT_ENFORCE_EQ(posix_memalign(&aligned_mem, align, sz), 0,
                "Cannot allocate align memory with %d bytes, "
                "align %d",
                sz, align);
  return aligned_mem;
}
} // namespace core
} // namespace fast_transformers
