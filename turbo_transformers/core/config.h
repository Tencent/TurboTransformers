

#pragma once
namespace turbo_transformers {
namespace core {
enum class BlasProvider { MKL, OpenBlas, BLIS };

BlasProvider GetBlasProvider();

void SetNumThreads(int n_th);

constexpr bool IsCompiledWithCUDA() {
#ifdef TT_WITH_CUDA
  return true;
#else
  return false;
#endif
}
}  // namespace core
}  // namespace turbo_transformers
