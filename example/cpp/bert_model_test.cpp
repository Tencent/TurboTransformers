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

#include "bert_model.h"

#include <cmath>
#include <future>
#include <thread>

int main() {
  bool use_cuda = false;
  BertModel model("models/bert.npz",
                  use_cuda ? DLDeviceType::kDLGPU : DLDeviceType::kDLCPU, 12,
                  12);
  std::vector<std::vector<int64_t>> position_ids{{1, 0, 0, 0}, {1, 1, 1, 0}};
  std::vector<std::vector<int64_t>> segment_ids{{1, 1, 1, 0}, {1, 0, 0, 0}};
  auto vec = model({{12166, 10699, 16752, 4454}, {5342, 16471, 817, 16022}},
                   position_ids, segment_ids, PoolType::kFirst, false);
  return 0;
}
