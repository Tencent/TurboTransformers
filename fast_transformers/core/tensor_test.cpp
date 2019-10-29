#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "fast_transformers/core/tensor.h"

TEST_CASE("TensorTest", "[init]") {
  REQUIRE_THROWS(FT_THROW("Test"));

  fast_transformers::core::Tensor test_tensor(fast_transformers::core::details::CreateDLPackTensor<float, ::fast_transformers::DeviceType::CPU>({3,4}));
  float* buff = test_tensor.mutableData<float>();
  for(int i = 0; i < 12; ++i)
    buff[i] = i * 0.1;
  for(int i = 0; i < 12; ++i)
    REQUIRE(fabs(test_tensor.data<float>()[i] - i * 0.1) < 1e-6);     

}

