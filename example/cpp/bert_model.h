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
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "dlpack/dlpack.h"
#include "turbo_transformers/layers/types.h"

using namespace turbo_transformers;
using PoolType = layers::types::PoolType;

class BertModel {
 public:
  BertModel(const std::string &filename, DLDeviceType device_type,
            size_t n_layers, int64_t n_heads);
  ~BertModel();

  std::vector<float> operator()(
      const std::vector<std::vector<int64_t>> &inputs,
      const std::vector<std::vector<int64_t>> &poistion_ids,
      const std::vector<std::vector<int64_t>> &segment_ids,
      PoolType pooling = PoolType::kFirst, bool use_pooler = false) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_;
};
