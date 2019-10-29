#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "fast_transformers/core/tensor.h"

TEST_CASE("TensorTest", "[tensor_allocate]") {
  REQUIRE_THROWS(FT_THROW("Test"));
  fast_transformers::core::Tensor test_tensor(fast_transformers::core::details::CreateDLPackTensor<float, ::fast_transformers::DeviceType::CPU>({3,4}));
  float* buff = test_tensor.mutableData<float>();
  for(int i = 0; i < 12; ++i)
    buff[i] = i * 0.1;
  for(int i = 0; i < 12; ++i)
    REQUIRE(fabs(test_tensor.data<float>()[i] - i*0.1) < 1e-6);
}

TEST_CASE("TensorTest2", "[tensor_init]") {
  fast_transformers::core::Tensor test_tensor(fast_transformers::core::details::CreateDLPackTensor<float, ::fast_transformers::DeviceType::CPU>({3,4}));
  test_tensor.Print<float>(std::cout);
  REQUIRE(test_tensor.n_dim() == 2);
  REQUIRE(test_tensor.numel() == 3*4);
}
