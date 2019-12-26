#define CATCH_CONFIG_MAIN
#ifdef FT_WITH_CUDA
#include "fast_transformers/core/cuda_device_context.h"
#endif
#include "catch2/catch.hpp"

namespace fast_transformers {
namespace core {

#ifdef FT_WITH_CUDA
TEST_CASE("CUDADeviceContext", "ini") {
  CUDADeviceContext& cuda_ctx = CUDADeviceContext::GetInstance();
  cuda_ctx.Wait();
}
#endif

}  // namespace core
}  // namespace fast_transformers
