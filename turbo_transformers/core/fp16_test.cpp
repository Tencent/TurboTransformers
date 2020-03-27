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

#include <fp16.h>

#include <cstdint>

#include "catch2/catch.hpp"
#include "turbo_transformers/core/half.h"
TEST_CASE("Half", "[Half]") {
  float input = 12.0f;
  turbo_transformers::core::Half half(input);
  CATCH_ENFORCE(input == static_cast<float>(half), "failed");
}
