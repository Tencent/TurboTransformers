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

#pragma once
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "turbo_transformers/core/allocator/model_aware_memory_scheduler.h"

namespace turbo_transformers {
namespace core {
namespace allocator {
namespace bert_config {

template <typename T>
void GetBertTensorUsageRecord(
    std::vector<TensorRecordItemPtr>& TensorUsageRecord,
    std::set<std::string>& activation_set, int64_t batch_size, int64_t seq_len,
    int64_t num_head = 12, int64_t hidden_size = 768, int64_t num_layer = 12);

}  // namespace bert_config
}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
