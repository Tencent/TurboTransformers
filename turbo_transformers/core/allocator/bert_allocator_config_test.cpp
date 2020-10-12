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

#include "turbo_transformers/core/allocator/bert_allocator_config.h"

#include <iostream>

#include "catch2/catch.hpp"
#include "turbo_transformers/core/allocator/chunked_allocator.h"

namespace turbo_transformers {
namespace core {
namespace allocator {

TEST_CASE("bert-allocator-test", "test1") {
  bert_config::GetBertTensorUsageRecord<float>(gBertTensorUsageRecord, 1, 40);
  ChunkedGreedyBySizeOffsetCalculation(gBertTensorUsageRecord,
                                       gTensorPositionMap);
  std::cerr << "gBertTensorUsageRecord " << gBertTensorUsageRecord.size()
            << std::endl;
  for (auto it = gTensorPositionMap.begin(); it != gTensorPositionMap.end();
       ++it) {
    std::cerr << it->first << " " << it->second.chunk_id_ << " "
              << it->second.offset_id_ << std::endl;
  }

  std::cerr << "TensorPositionMap " << gTensorPositionMap.size() << std::endl;
}

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
