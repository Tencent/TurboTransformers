#pragma once
#include <memory>

namespace fast_transformers {
namespace core {

static constexpr size_t gAlignment = 64U;

template <typename T>
class AlignedScratchpad {
 public:
  AlignedScratchpad();
  ~AlignedScratchpad();

  T* mutable_data(size_t numel);

  size_t capacity() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_;
};

}  // namespace core
}  // namespace fast_transformers
