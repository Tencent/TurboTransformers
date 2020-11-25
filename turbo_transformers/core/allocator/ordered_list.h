

#pragma once
#include <functional>
#include <memory>

namespace turbo_transformers {
namespace core {
namespace allocator {

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

  // use visitor to judge if the node should be deleted.
  void FreeNode(std::function<bool(T& node)> visitor) {
    Node* prev_node = GetHeadPtr();
    Node* cursor = head_ptr_->next_.get();
    while (cursor != nullptr) {
      // descending order
      auto next_node = cursor->next_.get();

      if (visitor(*cursor->content_)) {
        prev_node->next_ = std::move(cursor->next_);
        capacity_--;
      } else {
        prev_node = cursor;
      }
      cursor = next_node;
    }
  }

  Node* GetHeadPtr() { return head_ptr_.get(); }

  // add a new node constructed from new_node_ptr, while maintain the order of
  // list return : address of newly allocated chunk
  void Add(std::shared_ptr<T> content_ptr, bool reverse = false) {
    Node* prev_node = GetHeadPtr();
    Node* cursor = head_ptr_->next_.get();
    while (cursor != nullptr) {
      // descending order
      if (reverse && *content_ptr > *cursor->content_) {
        break;
        // ascending order
      } else if (!reverse && *content_ptr < *cursor->content_) {
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

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
