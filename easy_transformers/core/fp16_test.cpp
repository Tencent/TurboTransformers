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
#include <fp16.h>

#include <cstdint>

#include "catch2/catch.hpp"
#include "easy_transformers/core/half.h"
namespace easy_transformers {
namespace core {
TEST_CASE("Half", "[Half]") {
  float input = 12.0f;
  Half half(input);
  CATCH_ENFORCE(input == float(half), "failed");
}
}  // namespace core
}  // namespace easy_transformers
