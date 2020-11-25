

#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif
#include "catch2/catch.hpp"

namespace turbo_transformers {
namespace core {

#ifdef TT_WITH_CUDA
TEST_CASE("CUDADeviceContext", "init") {
  CUDADeviceContext& cuda_ctx = CUDADeviceContext::GetInstance();
  cuda_ctx.Wait();
}

#endif

}  // namespace core
}  // namespace turbo_transformers
