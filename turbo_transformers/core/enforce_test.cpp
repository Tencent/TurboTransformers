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
#include "turbo_transformers/core/enforce.h"

#include "catch2/catch.hpp"
TEST_CASE("Enforce", "[enforce]") {
  bool ok = false;
  try {
    TT_ENFORCE(false, "test");
  } catch (turbo_transformers::core::details::EnforceNotMet &) {
    ok = true;
  }
  REQUIRE(ok);

  ok = false;
  try {
    TT_THROW("Test");
  } catch (turbo_transformers::core::details::EnforceNotMet &) {
    ok = true;
  }
  REQUIRE(ok);
}
