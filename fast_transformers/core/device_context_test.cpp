#define CATCH_CONFIG_MAIN
#ifdef FT_WITH_CUDA
#include "fast_transformers/core/cuda_device_context.h"
#endif
#include "catch2/catch.hpp"

namespace fast_transformers {
namespace core {

#ifdef FT_WITH_CUDA
TEST_CASE("CUDADeviceContext", "init") {
  CUDADeviceContext& cuda_ctx = CUDADeviceContext::GetInstance();
  cuda_ctx.Wait();
}

TEST_CASE("allocate and free") {
  CUDADeviceContext& cuda_ctx = CUDADeviceContext::GetInstance();
  void* device_mem = cuda_ctx.allocate(20 * 1e9);
  cuda_ctx.free(device_mem, 20 * 1e9);
  cuda_ctx.Wait();
}

TEST_CASE("allocate too much") {
  CUDADeviceContext& cuda_ctx = CUDADeviceContext::GetInstance();
  void* device_mem1 = cuda_ctx.allocate(10 * 1e9);
  void* device_mem2 = cuda_ctx.allocate(10 * 1e9);
  // allocate 30 GB
  cuda_ctx.free(device_mem1, 10 * 1e9);
  void* device_mem3 = cuda_ctx.allocate(10 * 1e9);
  cuda_ctx.free(device_mem2, 10 * 1e9);
  cuda_ctx.free(device_mem3, 10 * 1e9);
  cuda_ctx.Wait();
}
#endif

}  // namespace core
}  // namespace fast_transformers
