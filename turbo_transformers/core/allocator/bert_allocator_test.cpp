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

struct TUR {
  std::string name_;
  int64_t start_op_;
  int64_t end_op_;
  int64_t size_;
};

static std::vector<TUR> tur_ref = {
    {"PrepareBertMasks/possitionids", 0, 1, 256},
    {"PrepareBertMasks/seqids/Reshape", 0, 1, 256},
    {"PrepareBertMasks/attmask/Reshape", 0, 1, 256},
    {"PrepareBertMasks/extendedattnmask/Reshape", 0, 1, 256},
    {"BERTEmbedding/Reshape", 1, 2, 122880},
    {"self/qkv_out1/Reshape", 0, 1, 368640},
    {"self/q/Reshape", 1, 2, 122880},
    {"self/k/Reshape", 1, 2, 122880},
    {"self/v/Reshape", 1, 3, 122880},
    {"batch_gemm3/Reshape", 2, 3, 76800},
    {"ApplyMaskAndSoftmax/Reshape", 3, 4, 122880},
    {"batch_gemm4/Reshape", 4, 5, 76800},
    {"gemm5/Reshape", 5, 8, 122880},
    {"BertIntermediate/Reshape", 7, 8, 491520},
    {"BertOutput/Reshape", 0, 9, 122880},

};

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

/*
TEST_CASE("chunk", "add a new tensor to a chunk") {
  ChunkList chunk_list([](size_t size) -> char*  { return new char[100]; });
  chunk_list.AddChunk(2 * 1024 *1024);
  chunk_list.ShowMe();

  chunk_list.visit([](Chunk* node) {
    std::shared_ptr<TensorRecordItem> t =
std::make_shared<TensorRecordItem>("tensor1", 1, 2, 100); node->AppendTensor(t,
0);
  });
  chunk_list.ShowMe();

  chunk_list.visit([](Chunk* node) {
    std::shared_ptr<TensorRecordItem> t =
std::make_shared<TensorRecordItem>("tensor2", 3, 4, 100); node->AppendTensor(t,
0);
  });
  chunk_list.ShowMe();

  chunk_list.visit([](Chunk* node) {
    std::shared_ptr<TensorRecordItem> t =
std::make_shared<TensorRecordItem>("tensor3", 3, 4, 100); node->AppendTensor(t,
200);

  });
  chunk_list.ShowMe();
}
*/

TEST_CASE("bert-config", "make sure generated bert tensor usage is correct") {
  std::vector<TensorRecordItemPtr> bert_tensor_usage_record;
  std::set<std::string> activation_set;
  bert_config::GetBertTensorUsageRecord<float>(
      bert_tensor_usage_record, activation_set, 1, 40, 12, 768, 12);

  for (auto it : bert_tensor_usage_record) {
    auto name = it->name_;
    auto start_op = it->start_op_;
    auto end_op = it->end_op_;
    auto size = it->size_;

    std::cerr << name << " " << size << std::endl;
    bool found{false};
    for (const auto& item : tur_ref) {
      if (item.name_ == name) {
        REQUIRE(start_op == item.start_op_);
        REQUIRE(end_op == item.end_op_);
        REQUIRE(size == item.size_);
        found = true;
      }
    }
    REQUIRE(found);
  }
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

  std::vector<int64_t> batch_list{1, 1, 2, 4, 1};
  std::vector<int64_t> seq_len_list{10, 100, 32, 500, 10};
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
