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
#include "fast_transformers/core/enforce.h"
#include "catch2/catch.hpp"
// TEST(enforce, throw) { ASSERT_THROW(WXBOT_THROW("Test"),
// wxbot::base::EnforceNotMet); } TEST(enforce, check) {
// ASSERT_THROW(WXBOT_ENFORCE(false, "Not matched"),
// wxbot::base::EnforceNotMet); ASSERT_THROW(WXBOT_ENFORCE_EQ(1, 2, "Not
// matched"), wxbot::base::EnforceNotMet); ASSERT_THROW(WXBOT_ENFORCE_NE(1, 1,
// "Matched"), wxbot::base::EnforceNotMet);
//}
TEST_CASE("Enforce", "[enforce]") {
  REQUIRE_THROWS(FT_THROW("Test"));
  bool ok = false;
  try {
    FT_ENFORCE(false, "test");
  } catch (fast_transformers::core::EnforceNotMet &) {
    ok = true;
  }
  REQUIRE(ok);
}
