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
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "loguru.hpp"
#include "turbo_transformers/core/allocator/ordered_list.h"

namespace turbo_transformers {
namespace core {
namespace allocator {

constexpr int64_t DEFAULT_TRUNK_SIZE = 2 * 1024 * 1024;
constexpr float K_SCALE = 1.2;

struct TensorRecordItem {
  TensorRecordItem(const std::string& name, int64_t start_op, int64_t end_op,
                   int64_t size)
      : name_(name), start_op_(start_op), end_op_(end_op), size_(size){};
  std::string name_;
  int64_t start_op_;
  int64_t end_op_;
  int64_t size_;
};
using TensorRecordItemPtr = std::shared_ptr<TensorRecordItem>;

class Chunk {
 public:
  Chunk(char* addr, int64_t size) : memaddr_(addr), size_(size) {}
  // a list of tensor usage info (name, first_op, last_op, size, offset)
  struct ChunkNode {
    ChunkNode(const TensorRecordItemPtr t, int64_t offset)
        : tensor_record_(t), offset_(offset) {}
    const TensorRecordItemPtr tensor_record_;
    int64_t offset_;
    bool operator<(const ChunkNode& o) const { return offset_ < o.offset_; }
    bool operator>(const ChunkNode& o) const { return offset_ > o.offset_; }
    bool operator>=(const ChunkNode& o) const { return offset_ >= o.offset_; }
    bool operator<=(const ChunkNode& o) const { return offset_ <= o.offset_; }
  };

  bool operator<(const Chunk& o) const { return size_ < o.size_; }
  bool operator>(const Chunk& o) const { return size_ > o.size_; }
  bool operator>=(const Chunk& o) const { return size_ >= o.size_; }
  bool operator<=(const Chunk& o) const { return size_ <= o.size_; }

  int64_t size() const { return size_; }

  void visit(std::function<void(ChunkNode*)> visitor) {
    tensor_info_.visit(visitor);
  }

  void AppendTensor(const TensorRecordItemPtr t, int64_t offset) {
    tensor_info_.Add(std::make_shared<ChunkNode>(t, offset));
  }

  void* GetMemAddr() const { return static_cast<void*>(memaddr_); }

  void showMe() {
    int64_t max_end_addr = 0;
    tensor_info_.visit([&](ChunkNode* node) {
      //            LOG_S(INFO) << node->tensor_record_->name_ << " "
      //                        << node->tensor_record_->size_ << " " <<
      //                        node->offset_;
      max_end_addr =
          std::max(max_end_addr, node->tensor_record_->size_ + node->offset_);
    });
    LOG_S(INFO) << "max end addr " << max_end_addr;
  }

  size_t TensorNum() const { return tensor_info_.capacity(); }

  // release all tensors
  void Reset() { tensor_info_.Reset(); }

 private:
  char* memaddr_{nullptr};
  OrderedList<ChunkNode> tensor_info_;
  int64_t size_;
};

class ChunkList {
 public:
  explicit ChunkList(std::function<char*(size_t)> mem_allocate_func,
                     std::function<void(void*)> mem_free_func)
      : mem_allocate_func_(mem_allocate_func), mem_free_func_(mem_free_func) {}

  // visitor visit the tensors of each chunk
  void visit(std::function<void(Chunk*)> visitor) {
    chunk_list_.visit(visitor);
  }

  // remove useless chunk in the list
  void Shrink() {
    chunk_list_.FreeNode([this](Chunk& chunk) -> bool {
      bool ret = (chunk.TensorNum() == 0);
      if (ret) {
        this->mem_free_func_(chunk.GetMemAddr());
      }
      return ret;
    });
  }

  Chunk* AddChunk(int64_t chunk_size) {
    char* addr = mem_allocate_func_(chunk_size);
    auto new_chunk = std::make_shared<Chunk>(addr, chunk_size);
    chunk_list_.Add(new_chunk);
    return back_ptr_ = new_chunk.get();
  }

  void ShowChunkUsage() {
    chunk_list_.visit([](Chunk* node) {
      LOG_S(INFO) << "tensor usage records in the chunk " << node->GetMemAddr()
                  << " of size " << node->size() << " B, #tensor "
                  << node->TensorNum();
      node->showMe();
    });
  }

  void Reset() {
    chunk_list_.visit([](Chunk* chunk) { chunk->Reset(); });
  }

 private:
  std::function<char*(size_t)> mem_allocate_func_;
  std::function<void(void*)> mem_free_func_;
  OrderedList<Chunk> chunk_list_{};
  Chunk* back_ptr_;
};

struct TensorPositionInfo {
  TensorPositionInfo(Chunk* chunk_ptr, int64_t offset)
      : chunk_ptr_(chunk_ptr), offset_(offset) {}
  Chunk* chunk_ptr_;
  int64_t offset_;
};

extern void ChunkedGreedyBySizeOffsetCalculation(
    const std::vector<TensorRecordItemPtr>& tensor_usage_record,
    ChunkList& chunk_list,
    std::map<std::string, TensorPositionInfo>& tensor_position_map);

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
