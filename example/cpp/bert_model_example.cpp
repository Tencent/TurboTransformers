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

#include "bert_model.h"

#include <cassert>
#include <cmath>
#include <future>
#include <iostream>
#include <thread>

#include "turbo_transformers/core/config.h"

static bool test(bool use_cuda = false) {
  // construct a bert model using n_layers and n_heads,
  // the hidden_size can be infered from the parameters
  BertModel model("models/bert.npz",
                  use_cuda ? DLDeviceType::kDLGPU : DLDeviceType::kDLCPU,
                  12, /* n_layers */
                  12 /* *n_heads */);
  std::vector<std::vector<int64_t>> position_ids{{1, 0, 0, 0}, {1, 1, 1, 0}};
  std::vector<std::vector<int64_t>> segment_ids{{1, 1, 1, 0}, {1, 0, 0, 0}};
  auto vec = model({{12166, 10699, 16752, 4454}, {5342, 16471, 817, 16022}},
                   position_ids, segment_ids, PoolType::kFirst,
                   true /* use a pooler after the encoder output */);

  assert(fabs(vec.data()[0] - 0.9151) < 1e-3);
  assert(fabs(vec.data()[1] - 0.5919) < 1e-3);
  assert(fabs(vec.data()[768] - 0.9802) < 1e-3);
  assert(fabs(vec.data()[768 + 1] - 0.9321) < 1e-3);

  return true;
}

static std::vector<float> CallBackFunction(
    const std::shared_ptr<BertModel> model,
    const std::vector<std::vector<int64_t>> input_ids,
    const std::vector<std::vector<int64_t>> position_ids,
    const std::vector<std::vector<int64_t>> segment_ids, PoolType pooltype,
    bool use_pooler) {
  return model->operator()(input_ids, position_ids, segment_ids, pooltype,
                           use_pooler);
}

bool test_multiple_threads(bool only_input, int n_threads) {
  std::shared_ptr<BertModel> model_ptr = std::make_shared<BertModel>(
      "models/bert.npz", DLDeviceType::kDLCPU, 12, 12);
  std::vector<std::vector<int64_t>> input_ids{{12166, 10699, 16752, 4454},
                                              {5342, 16471, 817, 16022}};
  std::vector<std::vector<int64_t>> position_ids{{1, 0, 0, 0}, {1, 1, 1, 0}};
  std::vector<std::vector<int64_t>> segment_ids{{1, 1, 1, 0}, {1, 0, 0, 0}};
  if (only_input) {
    position_ids.clear();
    segment_ids.clear();
  }
  std::vector<std::thread> threads;
  threads.reserve(n_threads);

  std::vector<std::future<std::vector<float>>> result_list;
  for (int i = 0; i < n_threads; ++i) {
    std::packaged_task<std::vector<float>(
        const std::shared_ptr<BertModel>,
        const std::vector<std::vector<int64_t>> &,
        const std::vector<std::vector<int64_t>> &,
        const std::vector<std::vector<int64_t>> &, PoolType, bool)>
        task(CallBackFunction);
    result_list.emplace_back(task.get_future());
    threads.emplace_back(std::thread(std::move(task), model_ptr, input_ids,
                                     position_ids, segment_ids,
                                     PoolType::kFirst, true));
  }

  for (int i = 0; i < n_threads; ++i) {
    auto vec = result_list[i].get();
    assert(vec.size() == 768 * 2);

    for (size_t i = 0; i < vec.size(); ++i) {
      assert(!std::isnan(vec.data()[i]));
      assert(!std::isinf(vec.data()[i]));
    }
    if (only_input) {
      assert(fabs(vec.data()[0] - 0.9671) < 1e-3);
      assert(fabs(vec.data()[1] - 0.9860) < 1e-3);
      assert(fabs(vec.data()[768] - 0.9757) < 1e-3);
      assert(fabs(vec.data()[768 + 1] - 0.9794) < 1e-3);
    } else {
      assert(fabs(vec.data()[0] - 0.9151) < 1e-3);
      assert(fabs(vec.data()[1] - 0.5919) < 1e-3);
      assert(fabs(vec.data()[768] - 0.9802) < 1e-3);
      assert(fabs(vec.data()[768 + 1] - 0.9321) < 1e-3);
    }
  }

  for (int i = 0; i < n_threads; ++i) {
    threads[i].join();
  }
  return true;
}

int main() {
  std::cout << "run bert on GPU, device id is 0" << std::endl;
  test(true);
  std::cout << "run bert on CPU, use 4 threads to do bert inference"
            << std::endl;
  turbo_transformers::core::SetNumThreads(4);
  test(false);
  std::cout << "10 threads do 10 independent bert inferences." << std::endl;
  turbo_transformers::core::SetNumThreads(1);
  test_multiple_threads(false, 10);
  test_multiple_threads(true, 10);
}
