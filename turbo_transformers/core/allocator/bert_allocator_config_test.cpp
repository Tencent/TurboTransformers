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

#include <iostream>

#include "catch2/catch.hpp"
#include "turbo_transformers/core/allocator/bert_config.h"
#include "turbo_transformers/core/allocator/chunked_allocator.h"

namespace turbo_transformers {
namespace core {
namespace allocator {

TEST_CASE("bert-allocator-test", "test1") {
  std::vector<TensorRecordItemPtr> bert_tensor_usage_record;
  bert_config::GetBertTensorUsageRecord<float>(bert_tensor_usage_record, 1, 40);
}

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
