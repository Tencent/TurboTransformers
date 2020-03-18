#define CATCH_CONFIG_MAIN
#include "turbo_transformers/loaders/modeling_bert.h"

#include "catch2/catch.hpp"
#include "turbo_transformers/core/macros.h"
namespace turbo_transformers {
namespace loaders {

TEST_CASE("Bert", "all") {
  BertModel model(
      "models/bert.npz",
      base::IsCompiledWithCUDA() ? DLDeviceType::kDLGPU : DLDeviceType::kDLCPU,
      12, 12);
  auto vec = model({{1, 2, 3, 4, 5}, {3, 4, 5}, {6, 7, 8, 9, 10, 11}},
                   PoolingType::kFirst);
  REQUIRE(vec.size() == 768 * 3);
}

}  // namespace loaders
}  // namespace turbo_transformers
