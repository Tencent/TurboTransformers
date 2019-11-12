#define CATCH_CONFIG_MAIN
#include "fast_transformers/core/aligned_scratchpad.h"
#include <stdint.h>
#include "catch2/catch.hpp"
namespace fast_transformers {
namespace core {
TEST_CASE("Enforce", "[enforce]") {
  AlignedScratchpad<float> buf;
  float* ptr1 = buf.mutable_data(10);
  REQUIRE(reinterpret_cast<uintptr_t>(ptr1) % gAlignment == 0);
  REQUIRE(buf.capacity() == 10);

  float* ptr2 = buf.mutable_data(20);
  REQUIRE(reinterpret_cast<uintptr_t>(ptr2) % gAlignment == 0);
  REQUIRE(buf.capacity() == 20);
  REQUIRE(ptr1 != ptr2);
  float* ptr3 = buf.mutable_data(15);
  REQUIRE(ptr2 == ptr3);
  REQUIRE(buf.capacity() == 20);
}

}  // namespace core
}  // namespace fast_transformers
