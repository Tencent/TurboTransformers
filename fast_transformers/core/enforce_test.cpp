#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "fast_transformers/core/enforce.h"
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
