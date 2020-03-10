// Copyright 2020 Tencent
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
