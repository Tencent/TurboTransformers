

#include <fp16.h>

#include <cstdint>

#include "catch2/catch.hpp"
#include "turbo_transformers/core/half.h"
TEST_CASE("Half", "[Half]") {
  float input = 12.0f;
  turbo_transformers::core::Half half(input);
  CATCH_ENFORCE(input == static_cast<float>(half), "failed");
}
