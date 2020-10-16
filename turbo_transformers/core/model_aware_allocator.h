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
#include <memory.h>

#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "macros.h"
#include "turbo_transformers/core/allocator/allocator_impl.h"

namespace turbo_transformers {
namespace core {

extern void schedule_dynamic_api(
    const std::unordered_map<std::string, int64_t> &assigned_offset,
    const std::unordered_map<std::string, int64_t> &assigned_trunk,
    const std::vector<int64_t> &trunk_info, const std::string &dev_str);

extern void bert_opt_mem_allocate_api(int64_t batch_size, int64_t seq_len,
                                      int64_t num_head, int64_t hidden_size,
                                      int64_t num_layer,
                                      const std::string &dev_str);

}  // namespace core
}  // namespace turbo_transformers
