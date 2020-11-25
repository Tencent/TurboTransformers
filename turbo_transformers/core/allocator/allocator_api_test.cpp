

#include "turbo_transformers/core/allocator/allocator_api.h"

#include <vector>

#include "catch2/catch.hpp"
#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace core {
namespace allocator {

TEST_CASE("naive-allocator-cpu") {
  Allocator &allocator = Allocator::GetInstance();
  std::vector<size_t> size_list{100, 100, 1000, 256, 200};
  std::vector<void *> addr_list(4);
  for (size_t i = 0; i < size_list.size(); ++i) {
    addr_list[i] = allocator.allocate(size_list[i], kDLCPU, "");
    allocator.free(addr_list[i], kDLCPU, "");
  }
}

TEST_CASE("model-aware-allocator-cpu-no-name") {
  Allocator &allocator = Allocator::GetInstance();
  allocator.set_schema("model-aware");
  std::vector<size_t> size_list{100, 100, 1000, 256, 200};
  std::vector<void *> addr_list(4);
  for (size_t i = 0; i < size_list.size(); ++i) {
    addr_list[i] = allocator.allocate(size_list[i], kDLCPU, "");
    allocator.free(addr_list[i], kDLCPU, "");
  }
  allocator.set_schema("naive");
}

TEST_CASE("model-aware-allocator-cpu-with-name") {
  Allocator &allocator = Allocator::GetInstance();
  allocator.set_schema("model-aware");

  std::vector<size_t> size_list{100, 100, 1000, 256, 200};
  std::vector<void *> addr_list(4);

  // batch size, seq_len, num_head, hidden_size, num_layer
  allocator.set_config({1, 40, 12, 768, 12});
  for (size_t i = 0; i < size_list.size(); ++i) {
    addr_list[i] =
        allocator.allocate(size_list[i], kDLCPU, "BertIntermediate/Reshape");
    allocator.free(addr_list[i], kDLCPU, "BertIntermediate/Reshape");
    REQUIRE(allocator.is_activation("BertIntermediate/Reshape"));
    REQUIRE(!allocator.is_activation("Reshape"));
  }
  allocator.set_schema("naive");
}

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
