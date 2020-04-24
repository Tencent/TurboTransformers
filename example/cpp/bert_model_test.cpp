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

#include "example/cpp/bert_model.h"
#include <cmath>
#include <future>
#include <thread>
#include <vector>

#include <iostream>
#include "catch2/catch.hpp"
#include "turbo_transformers/core/config.h"
#include "turbo_transformers/core/macros.h"

namespace turbo_transformers {
namespace loaders {
bool CheckCppBert(bool use_cuda, bool only_input) {
  BertModel model(model_file_path,
                  use_cuda ? DLDeviceType::kDLGPU : DLDeviceType::kDLCPU, 12,
                  12);
  std::vector<std::vector<int64_t>> position_ids{{1, 0, 0, 0}, {1, 1, 1, 0}};
  std::vector<std::vector<int64_t>> segment_ids{{1, 1, 1, 0}, {1, 0, 0, 0}};
  if (only_input) {
    position_ids.clear();
    segment_ids.clear();
  }
  auto vec = model({{12166, 10699, 16752, 4454}, {5342, 16471, 817, 16022}},
                   position_ids, segment_ids, PoolType::kFirst, false);
  REQUIRE(vec.size() == 768 * 2);
  // Write a better UT
  for (size_t i = 0; i < vec.size(); ++i) {
    REQUIRE(!std::isnan(vec.data()[i]));
    REQUIRE(!std::isinf(vec.data()[i]));
  }
  if (only_input) {
    std::cerr << vec.data()[0] << std::endl;
    REQUIRE(fabs(vec.data()[0] - -0.9791) < 1e-3);
    REQUIRE(fabs(vec.data()[1] - 0.8283) < 1e-3);
    REQUIRE(fabs(vec.data()[768] - 0.3837) < 1e-3);
    REQUIRE(fabs(vec.data()[768 + 1] - 0.1659) < 1e-3);
  } else {
    REQUIRE(fabs(vec.data()[0] - -0.3760) < 1e-3);
    REQUIRE(fabs(vec.data()[1] - 0.5674) < 1e-3);
    REQUIRE(fabs(vec.data()[768] - -0.2701) < 1e-3);
    REQUIRE(fabs(vec.data()[768 + 1] - 0.2676) < 1e-3);
  }
  return true;
}

bool CheckCppBertWithPooler(bool use_cuda, bool only_input) {
  BertModel model("models/bert.npz",
                  use_cuda ? DLDeviceType::kDLGPU : DLDeviceType::kDLCPU, 12,
                  12);
  std::vector<std::vector<int64_t>> position_ids{{1, 0, 0, 0}, {1, 1, 1, 0}};
  std::vector<std::vector<int64_t>> segment_ids{{1, 1, 1, 0}, {1, 0, 0, 0}};
  if (only_input) {
    position_ids.clear();
    segment_ids.clear();
  }
  auto vec = model({{12166, 10699, 16752, 4454}, {5342, 16471, 817, 16022}},
                   position_ids, segment_ids, PoolType::kFirst,
                   /*use_pooler*/ true);
  REQUIRE(vec.size() == 768 * 2);
  // Write a better UT
  for (size_t i = 0; i < vec.size(); ++i) {
    REQUIRE(!std::isnan(vec.data()[i]));
    REQUIRE(!std::isinf(vec.data()[i]));
  }
  if (only_input) {
    std::cerr << vec.data()[0] << std::endl;
    REQUIRE(fabs(vec.data()[0] - 0.9671) < 1e-3);
    REQUIRE(fabs(vec.data()[1] - 0.9860) < 1e-3);
    REQUIRE(fabs(vec.data()[768] - 0.9757) < 1e-3);
    REQUIRE(fabs(vec.data()[768 + 1] - 0.9794) < 1e-3);
  } else {
    REQUIRE(fabs(vec.data()[0] - 0.9151) < 1e-3);
    REQUIRE(fabs(vec.data()[1] - 0.5919) < 1e-3);
    REQUIRE(fabs(vec.data()[768] - 0.9802) < 1e-3);
    REQUIRE(fabs(vec.data()[768 + 1] - 0.9321) < 1e-3);
  }
  return true;
}

TEST_CASE("Bert", "Cpp interface") {
  CheckCppBert(false /*use_cuda*/, true /* only_input*/);
  CheckCppBert(false /*use_cuda*/, false /* only_input*/);
  if (core::IsCompiledWithCUDA()) {
    CheckCppBert(true /*use_cuda*/, true /* only_input*/);
    CheckCppBert(true /*use_cuda*/, false /* only_input*/);
  }
}

TEST_CASE("BertWithPooler", "Cpp interface") {
  CheckCppBertWithPooler(false /*use_cuda*/, false /* only_input*/);
  CheckCppBertWithPooler(false /*use_cuda*/, true /* only_input*/);
  if (core::IsCompiledWithCUDA()) {
    CheckCppBertWithPooler(true /*use_cuda*/, false /* only_input*/);
    CheckCppBertWithPooler(true /*use_cuda*/, true /* only_input*/);
  }
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

static bool test_multiple_threads(bool only_input, int n_threads) {
  std::shared_ptr<BertModel> model_ptr = std::make_shared<BertModel>(
      model_file_path, DLDeviceType::kDLCPU, 12, 12);
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
  result_list.reserve(n_threads);
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
TEST_CASE("Bert-multiple-thread", "Cpp interface") {
  test_multiple_threads(false, 10);
  test_multiple_threads(true, 10);
}

}  // namespace loaders
}  // namespace turbo_transformers
