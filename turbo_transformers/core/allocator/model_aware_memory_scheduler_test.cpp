

#include "turbo_transformers/core/allocator/model_aware_memory_scheduler.h"

#include <iostream>

#include "catch2/catch.hpp"

namespace turbo_transformers {
namespace core {
namespace allocator {

TEST_CASE("ordered_list1", "insert1") {
  OrderedList<int> ordered_list;

  std::shared_ptr<int> new_node_ptr1(new int(2));
  ordered_list.AddAfter(new_node_ptr1, ordered_list.GetHeadPtr());
  std::shared_ptr<int> new_node_ptr2(new int(1));
  ordered_list.AddAfter(new_node_ptr2, ordered_list.GetHeadPtr());

  ordered_list.Add(std::make_shared<int>(20));
  ordered_list.Add(std::make_shared<int>(5));
  ordered_list.Add(std::make_shared<int>(30));
  // a duplicated node
  ordered_list.Add(std::make_shared<int>(5));

  std::vector<int> ref{1, 2, 5, 5, 20, 30};
  size_t ref_idx{0};
  ordered_list.visit([&](int* node) {
    auto content = *node;
    //    std::cerr << content << std::endl;
    REQUIRE(content == ref[ref_idx++]);
  });
}

TEST_CASE("ordered_list2", "insert2") {
  OrderedList<int> ordered_list;
  int test_times = 5;
  while (test_times--) {
    ordered_list.Reset();
    bool reverse_flag = true;
    ordered_list.Add(std::make_shared<int>(20), reverse_flag);
    ordered_list.Add(std::make_shared<int>(5), reverse_flag);
    ordered_list.Add(std::make_shared<int>(30), reverse_flag);
    // a duplicated node
    ordered_list.Add(std::make_shared<int>(5), reverse_flag);

    std::vector<int> ref{30, 20, 5, 5};
    size_t ref_idx{0};
    ordered_list.visit([&](int* node) {
      auto content = *node;
      REQUIRE(content == ref[ref_idx++]);
    });
  }
}

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
