#pragma once

namespace fast_transformers {

namespace core {
enum class BlasProvider {
  MKL,
  OpenBlas,
};

extern bool IsWithCUDA();
extern BlasProvider GetBlasProvider();

}  // namespace core
}  // namespace fast_transformers
