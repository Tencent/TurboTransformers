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
#include "turbo_transformers/core/allocator/chunked_allocator.h"

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

TEST_CASE("bert-config", "test1") {
  std::vector<TensorRecordItemPtr> bert_tensor_usage_record;
  bert_config::GetBertTensorUsageRecord<float>(bert_tensor_usage_record, 1, 40);

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

TEST_CASE("bert-allocator", "test1") {
  std::vector<TensorRecordItemPtr> bert_tensor_usage_record;
  bert_config::GetBertTensorUsageRecord<float>(bert_tensor_usage_record, 1, 40);
  // TODO(jiaruifang) use a dummy allocator function.
  ChunkList chunk_list([](size_t size) -> char* { return new char[1]; });
  std::map<std::string, TensorPositionInfo> tensor_position_map;

  ChunkedGreedyBySizeOffsetCalculation(bert_tensor_usage_record, chunk_list,
                                       tensor_position_map);

  //  for (auto it : bert_tensor_usage_record) {
  //    auto name = it->name_;
  //    auto tmp = tensor_position_map.find(name);
  //    if (tmp == tensor_position_map.end()) REQUIRE(false);
  //    std::cerr << it->name_ << " " << it->size_ << " " <<
  //    tmp->second.chunk_ptr_
  //              << " " << tmp->second.offset_ << std::endl;
  //  }

  // tensor name, offset
  std::map<std::string, int64_t> ref = {
      {"BertIntermediate/Reshape", 0},
      {"self/qkv_out1/Reshape", 0},
      {"BERTEmbedding/Reshape", 368640},
      {"self/q/Reshape", 491520},
      {"self/k/Reshape", 614400},
      {"self/v/Reshape", 737280},
      {"ApplyMaskAndSoftmax/Reshape", 0},
      {"gemm5/Reshape", 491520},
      {"BertOutput/Reshape", 860160},
      {"batch_gemm3/Reshape", 122880},
      {"batch_gemm4/Reshape", 614400},
      {"PrepareBertMasks/possitionids", 983040},
      {"PrepareBertMasks/seqids/Reshape", 983296},
      {"PrepareBertMasks/attmask/Reshape", 983552},
      {"PrepareBertMasks/extendedattnmask/Reshape", 983808},
  };

  std::cerr << "name \t tensor_size \t offset" << std::endl;
  for (auto item : ref) {
    auto name = item.first;
    auto tmp = tensor_position_map.find(name);
    if (tmp == tensor_position_map.end()) REQUIRE(false);
    auto ref_offset = item.second;
    std::cerr << name << " " << tmp->second.offset_ << std::endl;
    REQUIRE(ref_offset == tmp->second.offset_);
    if (!(ref_offset == tmp->second.offset_))
      std::cerr << ref_offset << " vs " << tmp->second.offset_ << std::endl;
  }
}

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
