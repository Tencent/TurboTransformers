// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.
#include "turbo_transformers/core/allocator/allocator_api.h"

#include <vector>

#include "catch2/catch.hpp"
#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace core {
namespace allocator {

TEST_CASE("naive-allocator-cpu") {
  Allocator &allocator = Allocator::GetInstance();
  std::vector<size_t> size_list{100, 100, 1000, 256, 200};
  std::vector<void *> addr_list(4);
  for (size_t i = 0; i < size_list.size(); ++i) {
    addr_list[i] = allocator.allocate(size_list[i], kDLCPU, "");
    allocator.free(addr_list[i], kDLCPU, "");
  }
}

TEST_CASE("model-aware-allocator-cpu-no-name") {
  Allocator &allocator = Allocator::GetInstance();
  allocator.set_schema("model-aware");
  std::vector<size_t> size_list{100, 100, 1000, 256, 200};
  std::vector<void *> addr_list(4);
  for (size_t i = 0; i < size_list.size(); ++i) {
    addr_list[i] = allocator.allocate(size_list[i], kDLCPU, "");
    allocator.free(addr_list[i], kDLCPU, "");
  }
  allocator.set_schema("naive");
}

TEST_CASE("model-aware-allocator-cpu-with-name") {
  Allocator &allocator = Allocator::GetInstance();
  allocator.set_schema("model-aware");

  std::vector<size_t> size_list{100, 100, 1000, 256, 200};
  std::vector<void *> addr_list(4);

  // batch size, seq_len, num_head, hidden_size, num_layer
  allocator.set_config({1, 40, 12, 768, 12});
  for (size_t i = 0; i < size_list.size(); ++i) {
    addr_list[i] =
        allocator.allocate(size_list[i], kDLCPU, "BertIntermediate/Reshape");
    allocator.free(addr_list[i], kDLCPU, "BertIntermediate/Reshape");
    REQUIRE(allocator.is_activation("BertIntermediate/Reshape"));
    REQUIRE(!allocator.is_activation("Reshape"));
  }
  allocator.set_schema("naive");
}

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
