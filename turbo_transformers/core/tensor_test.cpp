

#include "turbo_transformers/core/tensor.h"

#include "catch2/catch.hpp"

namespace turbo_transformers {
namespace core {

TEST_CASE("TensorTest", "[tensor_allocate]") {
  turbo_transformers::core::Tensor test_tensor(
      turbo_transformers::core::NewDLPackTensorT<float>({3, 4}));
  float *buff = test_tensor.mutableData<float>();
  for (int i = 0; i < 12; ++i) buff[i] = i * 0.1;
  for (int i = 0; i < 12; ++i)
    REQUIRE(fabs(test_tensor.data<float>()[i] - i * 0.1) < 1e-6);
}

TEST_CASE("TensorTest2", "[tensor_init]") {
  turbo_transformers::core::Tensor test_tensor(
      turbo_transformers::core::NewDLPackTensorT<float>({3, 4}));
  test_tensor.Print<float>(std::cout);
  REQUIRE(test_tensor.n_dim() == 2);
  REQUIRE(test_tensor.numel() == 3 * 4);
}

#ifdef TT_WITH_CUDA
template <typename T>
inline void Fill(Tensor &tensor) {
  T *gpu_data = tensor.mutableData<T>();
  auto size = tensor.numel();
  std::unique_ptr<T[]> cpu_data(new T[size]);
  srand((unsigned)time(NULL));
  for (int64_t i = 0; i < size; ++i) {
    cpu_data[i] = static_cast<T>(rand() / RAND_MAX);
  }
  Memcpy(gpu_data, cpu_data.get(), size * sizeof(T), MemcpyFlag::kCPU2GPU);
}

TEST_CASE("TensorTest3", "GPU init") {
  turbo_transformers::core::Tensor test_tensor(
      turbo_transformers::core::NewDLPackTensorT<float>({3, 4}, kDLGPU, 0));
  Fill<float>(test_tensor);
  test_tensor.Print<float>(std::cout);
  REQUIRE(test_tensor.n_dim() == 2);
  REQUIRE(test_tensor.numel() == 3 * 4);
}
#endif

}  // namespace core
}  // namespace turbo_transformers
