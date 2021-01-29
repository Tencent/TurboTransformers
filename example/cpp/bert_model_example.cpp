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

#include <cassert>
#include <cmath>
#include <future>
#include <iostream>
#include <string>
#include <thread>

#include "bert_model.h"
#include "turbo_transformers/core/allocator/allocator_api.h"
#include "turbo_transformers/core/config.h"

static bool test_bert(const std::string &model_path, bool use_cuda = false) {
  // construct a bert model using n_layers and n_heads,
  // the hidden_size can be infered from the parameters

  BertModel model(model_path,
                  use_cuda ? DLDeviceType::kDLGPU : DLDeviceType::kDLCPU,
                  12, /* n_layers */
                  12 /* *n_heads */);
  std::vector<std::vector<int64_t>> position_ids{{1, 0, 0, 0}, {1, 1, 1, 0}};
  std::vector<std::vector<int64_t>> segment_ids{{1, 1, 1, 0}, {1, 0, 0, 0}};
  auto vec = model({{12166, 10699, 16752, 4454}, {5342, 16471, 817, 16022}},
                   position_ids, segment_ids, PoolType::kFirst,
                   true /* use a pooler after the encoder output */);
  // bert-base-uncased (2020.04.23 version), you may need to change it to
  assert(fabs(vec.data()[0] - -0.5503) < 1e-3);
  assert(fabs(vec.data()[1] - 0.1295) < 1e-3);
  assert(fabs(vec.data()[768] - -0.5545) < 1e-3);
  assert(fabs(vec.data()[768 + 1] - -0.1182) < 1e-3);

  return true;
}

static bool test_memory_opt_bert(const std::string &model_path,
                                 bool use_cuda = false) {
  // construct a bert model using n_layers and n_heads,
  // the hidden_size can be infered from the parameters

  auto &allocator =
      turbo_transformers::core::allocator::Allocator::GetInstance();
  allocator.set_config({2, 4, 12, 768, 12});

  BertModel model(model_path,
                  use_cuda ? DLDeviceType::kDLGPU : DLDeviceType::kDLCPU,
                  12, /* n_layers */
                  12 /* *n_heads */);
  std::vector<std::vector<int64_t>> position_ids{{1, 0, 0, 0}, {1, 1, 1, 0}};
  std::vector<std::vector<int64_t>> segment_ids{{1, 1, 1, 0}, {1, 0, 0, 0}};
  auto vec = model({{12166, 10699, 16752, 4454}, {5342, 16471, 817, 16022}},
                   position_ids, segment_ids, PoolType::kFirst,
                   true /* use a pooler after the encoder output */);
  // bert-base-uncased (2020.04.23 version), you may need to change it to
  assert(fabs(vec.data()[0] - -0.5503) < 1e-3);
  assert(fabs(vec.data()[1] - 0.1295) < 1e-3);
  assert(fabs(vec.data()[768] - -0.5545) < 1e-3);
  assert(fabs(vec.data()[768 + 1] - -0.1182) < 1e-3);

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

bool test_multiple_threads(const std::string &model_path, bool only_input,
                           bool use_cuda, int n_threads) {
  std::shared_ptr<BertModel> model_ptr =
      std::make_shared<BertModel>(model_path, DLDeviceType::kDLCPU, 12, 12);
  // input_ids, position_ids, segment_ids lengths of each row may not be the
  // same. For example. std::vector<std::vector<int64_t>> input_ids{{1, 2, 3, 4,
  // 5,  6, 7},
  //                                             {1, 2}};
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
    // Attention, the hard code value is based on huggingface/transformers
    // bert-base-uncased (2020.04.23 version), you may need to change it to
    // real-time values.
    if (only_input) {
      std::cerr << vec.data()[0] << std::endl;
      std::cerr << vec.data()[1] << std::endl;
      std::cerr << vec.data()[768] << std::endl;
      std::cerr << vec.data()[768 + 1] << std::endl;
      assert(fabs(vec.data()[0] - -0.1901) < 1e-3);
      assert(fabs(vec.data()[1] - 0.0193) < 1e-3);
      assert(fabs(vec.data()[768] - 0.3060) < 1e-3);
      assert(fabs(vec.data()[768 + 1] - 0.1162) < 1e-3);
    } else {
      std::cerr << vec.data()[0] << std::endl;
      std::cerr << vec.data()[1] << std::endl;
      std::cerr << vec.data()[768] << std::endl;
      std::cerr << vec.data()[768 + 1] << std::endl;
      assert(fabs(vec.data()[0] - -0.5503) < 1e-3);
      assert(fabs(vec.data()[1] - 0.1295) < 1e-3);
      assert(fabs(vec.data()[768] - -0.5545) < 1e-3);
      assert(fabs(vec.data()[768 + 1] - -0.1182) < 1e-3);
    }
  }

  for (int i = 0; i < n_threads; ++i) {
    threads[i].join();
  }
  return true;
}

using namespace turbo_transformers;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "./bert_example npz_model_path" << std::endl;
    return -1;
  }
  const std::string model_path = static_cast<std::string>(argv[1]);

  if (core::IsCompiledWithCUDA()) {
    std::cout << "run bert on GPU, device id is 0" << std::endl;
    // Test model-aware Allocator.
    // NOTE, if using the model-aware allocator,
    // then you shall not run multi bert inference concurrently.
    // Because all activations of the bert share the same memory space.
    auto &allocator =
        turbo_transformers::core::allocator::Allocator::GetInstance();

    allocator.set_schema("model-aware");
    test_memory_opt_bert(model_path, true /*use cuda*/);
    allocator.set_schema("naive");

    test_bert(model_path, true /*use cuda*/);
  }
  std::cout << "run bert on CPU, use 4 threads to do bert inference"
            << std::endl;
  turbo_transformers::core::SetNumThreads(4);
  test_bert(model_path, false /*not use cuda*/);
  turbo_transformers::core::SetNumThreads(1);
  if (core::IsCompiledWithCUDA()) {
    std::cout << "10 threads do 10 independent bert inferences." << std::endl;
    test_multiple_threads(model_path, false /*only_input*/, true /*use cuda*/,
                          10);
  }

  test_multiple_threads(model_path, false /*only_input*/,
                        false /*not use cuda*/, 1);

  return 0;
}
