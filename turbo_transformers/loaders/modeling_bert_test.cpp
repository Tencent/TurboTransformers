// Copyright 2020 Tencent
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define CATCH_CONFIG_MAIN
#include "turbo_transformers/loaders/modeling_bert.h"

#include "catch2/catch.hpp"
#include "turbo_transformers/core/macros.h"
namespace turbo_transformers {
namespace loaders {

TEST_CASE("Bert", "all") {
  BertModel model(
      "models/bert.npz",
      core::IsCompiledWithCUDA() ? DLDeviceType::kDLGPU : DLDeviceType::kDLCPU,
      12, 12);
  auto vec = model({{1, 2, 3, 4, 5}, {3, 4, 5}, {6, 7, 8, 9, 10, 11}},
                   PoolingType::kFirst);
  REQUIRE(vec.size() == 768 * 3);
}

}  // namespace loaders
}  // namespace turbo_transformers
