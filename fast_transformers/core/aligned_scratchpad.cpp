// Copyright 2020 Tencent
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
