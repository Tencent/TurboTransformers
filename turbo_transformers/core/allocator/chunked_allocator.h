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
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

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

template <typename T>
class OrderedList {
 public:
  OrderedList() : head_ptr_(new Node(nullptr, nullptr)) {}
  struct Node {
    Node(std::shared_ptr<T> ptr, std::unique_ptr<Node> next)
        : ptr_(ptr), next_(std::move(next)) {}
    std::shared_ptr<T> ptr_;
    std::unique_ptr<Node> next_;
  };

  Node* GetHeadPtr() { return head_ptr_.get(); }

  // O(N)
  void AddAfter(std::shared_ptr<T> new_node_ptr, Node* prev_node) {
    std::unique_ptr<Node> tmp(
        new Node(new_node_ptr, std::move(prev_node->next_)));
    prev_node->next_ = std::move(tmp);
  }

  // O(N)
  template <typename Visitor>
  void visit(Visitor visitor) {
    Node* cursor = head_ptr_->next_.get();
    while (cursor != nullptr) {
      visitor(*cursor->ptr_);
      cursor = cursor->next_.get();
    }
  }

 private:
  std::unique_ptr<Node> head_ptr_;
};

struct Chunk {
  explicit Chunk(int64_t size, int64_t id) : size_(size), id_(id) {}
  // a list of tensor usage info (name, first_op, last_op, size, offset)
  struct Node {
    Node(const TensorRecordItem& t, int64_t offset)
        : tensor_record_(t), offset_(offset) {}
    TensorRecordItem tensor_record_;
    int64_t offset_;
    bool operator<(Node& o) const { return offset_ < o.offset_; }
  };
  std::vector<Node> tensor_info_;
  int64_t size_;
  int64_t id_;

  template <typename Visitor>
  void visit(Visitor visitor) {
    for (auto t : tensor_info_) {
      visitor(t);
    }
  }

  void Sort() { std::sort(tensor_info_.begin(), tensor_info_.end()); }

  void AppendTensor(const TensorRecordItem& t, int64_t offset) {
    tensor_info_.emplace_back(std::move(Node(t, offset)));
  }
};

struct ChunkList {
  std::vector<Chunk> chunk_list_{};
  int size_{0};

  template <typename Visitor>
  void visit(Visitor visitor) {
    for (auto chunk : chunk_list_) {
      visitor(chunk);
    }
  }

  int64_t AppendChunk(int64_t chunk_size) {
    // TODO(jiaruifang) Allocate memory here
    std::cerr << "AppendChunk chunk " << size_ << std::endl;
    chunk_list_.push_back(Chunk(chunk_size, size_));
    size_++;
    return size_ - 1;
  }
};

struct TensorPositionInfo {
  TensorPositionInfo(int64_t chunk_id, int64_t offset_id)
      : chunk_id_(chunk_id), offset_id_(offset_id) {}
  int64_t chunk_id_;
  int64_t offset_id_;
};

static std::vector<TensorRecordItem> gBertTensorUsageRecord;
static ChunkList gChunkList;
static std::map<std::string, TensorPositionInfo> gTensorPositionMap;

extern void ChunkedGreedyBySizeOffsetCalculation(
    const std::vector<TensorRecordItem>& tensor_usage_record,
    std::map<std::string, TensorPositionInfo>& TensorPositionMap);

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
