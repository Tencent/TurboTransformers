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

#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "dlpack/dlpack.h"
namespace turbo_transformers {
namespace loaders {

enum class PoolingType {
  kMax = 0,
  kMean,
  kFirst,
  kLast,

};

class BertModel {
 public:
  BertModel(const std::string &filename, DLDeviceType device_type,
            size_t n_layers, int64_t n_heads);
  ~BertModel();

  std::vector<float> operator()(const std::vector<std::vector<int64_t>> &inputs,
                                PoolingType pooling) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_;
};

}  // namespace loaders
}  // namespace turbo_transformers
