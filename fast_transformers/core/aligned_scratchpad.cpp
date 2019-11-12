#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/enforce.h"
#include "fast_transformers/core/memory.h"
namespace fast_transformers {
namespace core {

struct CPointerDeleter {
 public:
  void operator()(void* ptr) const { free(ptr); }
};

template <typename T>
struct AlignedScratchpad<T>::Impl {
 public:
  std::unique_ptr<T, CPointerDeleter> buf_;
  size_t capacity_{0};
};

template <typename T>
AlignedScratchpad<T>::AlignedScratchpad() : m_(new Impl()) {}

template <typename T>
T* AlignedScratchpad<T>::mutable_data(size_t numel) {
  if (FT_UNLIKELY(m_->capacity_ < numel)) {
    m_->buf_.reset(align_alloc_t<T>(numel, gAlignment));
    m_->capacity_ = numel;
  }
  return m_->buf_.get();
}

template <typename T>
size_t AlignedScratchpad<T>::capacity() const {
  return m_->capacity_;
}

template <typename T>
AlignedScratchpad<T>::~AlignedScratchpad() = default;

template class AlignedScratchpad<float>;

}  // namespace core
}  // namespace fast_transformers
