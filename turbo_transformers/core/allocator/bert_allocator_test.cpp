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

#include "allocator_api.h"
#include "catch2/catch.hpp"
#include "turbo_transformers/core/allocator/bert_config.h"
#include "turbo_transformers/core/allocator/model_aware_memory_scheduler.h"

namespace turbo_transformers {
namespace core {
namespace allocator {

/***
 * Is a tensor_position_map valid
 * @param tensor_position_map, the allocation schema
 * @param tensor_usage_record, the usage
 * @return
 */
static bool CheckValid(
    std::map<std::string, TensorPositionInfo>& tensor_position_map,
    std::vector<TensorRecordItemPtr>& tensor_usage_record) {
  for (auto tensor : tensor_usage_record) {
    auto name = tensor->name_;
    auto start_op = tensor->start_op_;
    auto end_op = tensor->end_op_;
    auto size = tensor->size_;
    auto it = tensor_position_map.find(name);
    if (it == tensor_position_map.end()) return false;
    auto offset = it->second.offset_;
    Chunk* chunk_addr = it->second.chunk_ptr_;
    bool flag{true};
    chunk_addr->visit([&](Chunk::ChunkNode* node) {
      if (!flag) return;
      auto x_name = node->tensor_record_->name_;
      if (x_name == name) return;
      auto x_offset = node->offset_;
      auto x_start_op = node->tensor_record_->start_op_;
      auto x_end_op = node->tensor_record_->end_op_;
      auto x_size = node->tensor_record_->size_;
      if (std::max(x_start_op, start_op) <= std::min(x_end_op, end_op)) {
        // time overlap as well space overlap
        if (std::max(x_offset, offset) <
            std::min(x_size + x_offset, size + offset)) {
          flag = false;
          std::cerr << x_name << " " << x_start_op << " " << x_end_op << " "
                    << x_offset << " " << x_size << std::endl;
          std::cerr << name << " " << start_op << " " << end_op << " " << offset
                    << " " << size << std::endl;
        }
      }
    });
    if (!flag) return false;
  }  // for
  return true;
}

TEST_CASE("bert-allocator-multiple-chunk",
          "check memory scheme in multi chunks scenarios") {
  std::vector<TensorRecordItemPtr> bert_tensor_usage_record;
  std::set<std::string> activation_set;
  bert_config::GetBertTensorUsageRecord<float>(
      bert_tensor_usage_record, activation_set, 1, 40, 12, 768, 12);
  ChunkList chunk_list([](size_t size) -> char* { return new char[size]; },
                       [](void* mem_addr) { free(mem_addr); });
  std::map<std::string, TensorPositionInfo> tensor_position_map;

  ChunkedGreedyBySizeOffsetCalculation(bert_tensor_usage_record, chunk_list,
                                       tensor_position_map);

  std::cerr << "bert-allocator-multiple-chunk" << std::endl;
  REQUIRE(CheckValid(tensor_position_map, bert_tensor_usage_record));
}

TEST_CASE("compare-mem-opt-allocator-and-cub",
          "check multi times memory allocation correction") {
  std::vector<TensorRecordItemPtr> bert_tensor_usage_record;
  std::set<std::string> activation_set;
  std::map<std::string, TensorPositionInfo> tensor_position_map;
  bert_config::GetBertTensorUsageRecord<float>(
      bert_tensor_usage_record, activation_set, 1, 40, 12, 768, 12);

  // warmup
  bert_opt_mem_allocate_api(1, 40, 12, 768, 12, "GPU");

  std::vector<int> len_list = {
      318, 357, 139, 464, 334, 165, 382, 176, 480, 287, 440, 14,  169, 244,
      320, 132, 429, 178, 247, 69,  422, 398, 451, 329, 45,  352, 455, 331,
      118, 344, 338, 310, 187, 303, 285, 356, 445, 23,  54,  18,  268, 237,
      13,  294, 87,  430, 81,  302, 11,  132, 150, 182, 370, 343, 220, 108,
      43,  57,  205, 260, 115, 266, 455, 124, 93,  161, 332, 53,  25,  300,
      340, 191, 249, 384, 75,  156, 417, 426, 471, 188, 420, 498, 298, 46,
      250, 480, 105, 237, 469, 137, 160, 356, 405, 114, 202, 18,  232, 206,
      169, 345, 101, 165, 441, 369, 482, 443, 160, 171, 401, 183, 411, 436,
      105, 476, 378, 30,  226, 184, 361, 58,  90,  224, 20,  436, 295, 47,
      476, 50,  56,  439, 391, 456, 366, 7,   286, 385, 244, 207, 171, 261,
      295, 430, 442, 346, 32,  403, 400, 53,  234, 467};
  auto& allocator = Allocator::GetInstance();

  std::vector<std::tuple<int, float, float>> res_map;
  for (int len : len_list) {
    allocator.set_schema("model-aware");
    auto start = std::chrono::system_clock::now();

    bert_opt_mem_allocate_api(1, len, 12, 768, 12, "GPU");

    for (int i = 0; i < 12; ++i) {
      for (auto& item : bert_tensor_usage_record) {
        auto name = item->name_;
        auto size = item->size_;
//    std::cerr << name << " " << size << std::endl;
#ifdef TT_WITH_CUDA
        allocator.allocate(size, kDLCPU, name);
#else
        allocator.allocate(size, kDLGPU, name);
#endif
      }
    }

    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto ma_elapsed_time = float(duration.count()) *
                           std::chrono::microseconds::period::num /
                           std::chrono::microseconds::period::den;

    bert_config::GetBertTensorUsageRecord<float>(
        bert_tensor_usage_record, activation_set, 1, len, 12, 768, 12);

    start = std::chrono::system_clock::now();

    allocator.set_schema("naive");
    std::map<std::string, void*> mem_buff;
    for (int i = 0; i < 12; ++i) {
      for (auto& item : bert_tensor_usage_record) {
        auto name = item->name_ + std::to_string(i);
        auto size = item->size_;
#ifdef TT_WITH_CUDA
        void* addr = allocator.allocate(size, kDLCPU, name);
#else
        void* addr = allocator.allocate(size, kDLGPU, name);
#endif
        mem_buff.emplace(name, addr);
      }
    }

    for (int i = 0; i < 12; ++i) {
      for (auto& item : bert_tensor_usage_record) {
        auto name = item->name_ + std::to_string(i);
#ifdef TT_WITH_CUDA
        allocator.free(mem_buff[name], kDLCPU, name);
#else
        allocator.free(mem_buff[name], kDLGPU, name);
#endif
      }
    }

    end = std::chrono::system_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto naive_elapsed_time = float(duration.count()) *
                              std::chrono::microseconds::period::num /
                              std::chrono::microseconds::period::den;
    //    std::cerr << ma_elapsed_time << ", " << naive_elapsed_time <<
    //    std::endl;

    res_map.emplace_back(len, ma_elapsed_time, naive_elapsed_time);
  }

  std::sort(res_map.begin(), res_map.end(),
            [](auto& a, auto& b) { return std::get<0>(a) < std::get<0>(b); });
  for (auto item : res_map) {
    std::cerr << std::get<0>(item) << " " << std::get<1>(item) << " "
              << std::get<2>(item) << std::endl;
  }
}

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
