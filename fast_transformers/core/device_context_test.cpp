#define CATCH_CONFIG_MAIN
#include "fast_transformers/core/device_context.h"
#include "catch2/catch.hpp"

namespace fast_transformers {
namespace core {

TEST_CASE("device_context", "init pool") {
  DeviceContextPool& pool = DeviceContextPool::Instance();
  auto cpu_ctx = pool.Get(kDLCPU);
  REQUIRE(cpu_ctx != nullptr);
  auto gpu_ctx = pool.Get(kDLGPU);
  REQUIRE(gpu_ctx != nullptr);
  REQUIRE(pool.size() == 2);
}

}  // namespace core
}  // namespace fast_transformers

