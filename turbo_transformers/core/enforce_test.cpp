

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
