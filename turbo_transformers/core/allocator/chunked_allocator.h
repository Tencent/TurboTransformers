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

template <typename T>
class OrderedList {
 public:
  struct Node {
    Node(std::shared_ptr<T> ptr, std::unique_ptr<Node> next)
        : content_(ptr), next_(std::move(next)) {}
    std::shared_ptr<T> content_;
    std::unique_ptr<Node> next_;
  };

  OrderedList() : head_ptr_(new Node(nullptr, nullptr)), capacity_(0) {}

  size_t capacity() const { return capacity_; }
  void Reset() {
    // cascadely release the node on list except the head node.
    head_ptr_->next_.reset();
    capacity_ = 0;
  }

  Node* GetHeadPtr() { return head_ptr_.get(); }

  // add a new node constructed from new_node_ptr, while maintain the order of
  // list return : address of newly allocated chunk
  void Add(std::shared_ptr<T> content_ptr, bool reverse = false) {
    Node* prev_node = GetHeadPtr();
    Node* cursor = head_ptr_->next_.get();
    while (cursor != nullptr) {
      // descending order
      if (reverse && *content_ptr >= *cursor->content_) {
        break;
        // ascending order
      } else if (!reverse && *content_ptr <= *cursor->content_) {
        break;
      }
      prev_node = cursor;
      cursor = cursor->next_.get();
    }
    AddAfter(content_ptr, prev_node);
  }

  // Add a node after prev_node, the node is constructed form new_node_ptr
  // time complexity O(N)
  void AddAfter(std::shared_ptr<T> content_ptr, Node* prev_node) {
    std::unique_ptr<Node> tmp(
        new Node(content_ptr, std::move(prev_node->next_)));
    prev_node->next_ = std::move(tmp);
    capacity_++;
  }

  // visit the list in O(N) visitor (T*)
  void visit(std::function<void(T*)> visitor) {
    Node* cursor = head_ptr_->next_.get();
    while (cursor != nullptr) {
      visitor(cursor->content_.get());
      cursor = cursor->next_.get();
    }
  }

 private:
  std::unique_ptr<Node> head_ptr_;
  size_t capacity_;
};

class Chunk {
 public:
  Chunk(char* addr, int64_t size, int64_t id)
      : memaddr_(addr), size_(size), id_(id) {}
  // a list of tensor usage info (name, first_op, last_op, size, offset)
  struct ChunkNode {
    ChunkNode(const TensorRecordItemPtr t, int64_t offset)
        : tensor_record_(t), offset_(offset) {}
    const TensorRecordItemPtr tensor_record_;
    int64_t offset_;
    bool operator<(const ChunkNode& o) const { return offset_ < o.offset_; }
    bool operator<=(const ChunkNode& o) const { return offset_ <= o.offset_; }
    bool operator>=(const ChunkNode& o) const { return offset_ >= o.offset_; }
  };

  bool operator<(const Chunk& o) const { return size_ < o.size_; }

  bool operator>=(const Chunk& o) const { return size_ >= o.size_; }

  bool operator<=(const Chunk& o) const { return size_ <= o.size_; }

  OrderedList<ChunkNode> tensor_info_;
  int64_t size_;
  int64_t id_;

  void visit(std::function<void(ChunkNode*)> visitor) {
    tensor_info_.visit(visitor);
  }

  void AppendTensor(const TensorRecordItemPtr t, int64_t offset) {
    tensor_info_.Add(std::make_shared<ChunkNode>(t, offset));
  }

  char* GetMemAddr() const { return memaddr_; }

 private:
  char* memaddr_{nullptr};
};

class ChunkList {
 public:
  explicit ChunkList(std::function<char*(size_t)> mem_allocate_func)
      : mem_allocate_func_(mem_allocate_func) {}

  // visitor visit the tensors of each chunk
  void visit(std::function<void(Chunk*)> visitor) {
    chunk_list_.visit(visitor);
  }

  // TODO(jiaruifang) remove useless chunk in the list
  void Shrink() {}

  Chunk* AddChunk(int64_t chunk_size) {
    char* addr = mem_allocate_func_(chunk_size);
    auto new_chunk =
        std::make_shared<Chunk>(addr, chunk_size, chunk_list_.capacity() + 1);
    chunk_list_.Add(new_chunk);
    return new_chunk.get();
  }

 private:
  std::function<char*(size_t)> mem_allocate_func_;
  OrderedList<Chunk> chunk_list_{};
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
