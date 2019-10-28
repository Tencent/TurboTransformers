#pragma once
#include "absl/debugging/stacktrace.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include <array>
#include <stdexcept>
#include <string>
namespace fast_transformers {
namespace core {
/**
 * Implement enforce macros.
 *
 * Will throw `EnforceNotMet` when enforce check failed. Unlike GLOG and
 * std::assert, it:
 *   1. let the program catch the exception and recover from bad state.
 *   2. enforce carry a rich information about call stack. It is useful for
 * debugging.
 */

static constexpr size_t kStackLimit = 20UL;
static constexpr size_t kStackSkipCount = 1UL;
class EnforceNotMet : public std::exception {
public:
  explicit EnforceNotMet(std::string msg) : msg_(std::move(msg)) {
    n_ = absl::GetStackTrace(stacks_.data(), stacks_.size(), kStackSkipCount);
  }

  const char *what() const noexcept override;

private:
  mutable std::string msg_;
  std::array<void *, kStackLimit> stacks_{};
  size_t n_;
  mutable bool stack_added_{false};
};

#if !defined(_WIN32)
#define FT_UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#else
#define FT_UNLIKELY(condition) (condition)
#endif

#define FT_THROW(...)                                                          \
  throw ::fast_transformers::core::EnforceNotMet(absl::StrFormat(__VA_ARGS__))
#define FT_ENFORCE(cond, ...)                                                  \
  do {                                                                         \
    if (FT_UNLIKELY(!(cond))) {                                                \
      throw ::fast_transformers::core::EnforceNotMet(                          \
          absl::StrCat("enforce error", #cond,                                 \
                       absl::StrFormat(" at %s:%d\n", __FILE__, __LINE__),     \
                       absl::StrFormat(__VA_ARGS__)));                         \
    }                                                                          \
                                                                               \
  } while (false)

#define FT_ENFORCE_EQ(a, b, ...) FT_ENFORCE((a) == (b), __VA_ARGS__)
#define FT_ENFORCE_NE(a, b, ...) FT_ENFORCE((a) != (b), __VA_ARGS__)
#define FT_ENFORCE_LT(a, b, ...) FT_ENFORCE((a) < (b), __VA_ARGS__)
#define FT_ENFORCE_LE(a, b, ...) FT_ENFORCE((a) <= (b), __VA_ARGS__)

} // namespace core
} // namespace fast_transformers
