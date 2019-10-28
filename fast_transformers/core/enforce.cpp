#include "enforce.h"
#include "absl/debugging/symbolize.h"
#include <sstream>
namespace fast_transformers {
namespace core {
static constexpr size_t gBufSize = 128;
static thread_local char gBuffer[gBufSize];
const char *EnforceNotMet::what() const noexcept {
  if (!stack_added_) {
    std::ostringstream sout;
    sout << msg_ << "\n";
    sout << "Callstack " << n_ << "\n";
    for (size_t i = 0; i < n_; ++i) {
      void *frame = stacks_[i];
      sout << "\t" << frame;
      if (absl::Symbolize(frame, gBuffer, gBufSize)) {
        sout << "\t" << gBuffer;
      }
      sout << "\n";
    }
    msg_ = sout.str();
    stack_added_ = true;
  }
  return msg_.c_str();
}

} // namespace core
} // namespace fast_transformers
