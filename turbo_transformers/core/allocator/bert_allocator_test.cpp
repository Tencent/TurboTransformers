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

TEST_CASE("bert-allocator-multiple-allocation",
          "check multi times memory allocation correction") {
  std::vector<TensorRecordItemPtr> bert_tensor_usage_record;
  std::map<std::string, TensorPositionInfo> tensor_position_map;
  ChunkList chunk_list([](size_t size) -> char* { return new char[size]; },
                       [](void* mem_addr) { free(mem_addr); });

  std::vector<int64_t> batch_list{2, 1, 2};
  std::vector<int64_t> seq_len_list{50, 100, 50};
  std::set<std::string> activation_set;
  for (size_t i = 0; i < batch_list.size(); ++i) {
    LOG_S(INFO) << "begin allocate for batch " << batch_list[i] << " seq_len "
                << seq_len_list[i];
    bert_config::GetBertTensorUsageRecord<float>(bert_tensor_usage_record,
                                                 activation_set, batch_list[i],
                                                 seq_len_list[i], 12, 768, 12);

    ChunkedGreedyBySizeOffsetCalculation(bert_tensor_usage_record, chunk_list,
                                         tensor_position_map);

    chunk_list.ShowChunkUsage();
    REQUIRE(CheckValid(tensor_position_map, bert_tensor_usage_record));
  }
}

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
