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
#include "easy_transformers/core/tensor.h"

#include "catch2/catch.hpp"

namespace easy_transformers {
namespace core {

TEST_CASE("TensorTest", "[tensor_allocate]") {
  easy_transformers::core::Tensor test_tensor(
      easy_transformers::core::NewDLPackTensorT<float>({3, 4}));
  float *buff = test_tensor.mutableData<float>();
  for (int i = 0; i < 12; ++i) buff[i] = i * 0.1;
  for (int i = 0; i < 12; ++i)
    REQUIRE(fabs(test_tensor.data<float>()[i] - i * 0.1) < 1e-6);
}

TEST_CASE("TensorTest2", "[tensor_init]") {
  easy_transformers::core::Tensor test_tensor(
      easy_transformers::core::NewDLPackTensorT<float>({3, 4}));
  test_tensor.Print<float>(std::cout);
  REQUIRE(test_tensor.n_dim() == 2);
  REQUIRE(test_tensor.numel() == 3 * 4);
}

#ifdef FT_WITH_CUDA
template <typename T>
inline void Fill(Tensor &tensor) {
  T *gpu_data = tensor.mutableData<T>();
  auto size = tensor.numel();
  std::unique_ptr<T[]> cpu_data(new T[size]);
  srand((unsigned)time(NULL));
  for (int64_t i = 0; i < size; ++i) {
    cpu_data[i] = rand() / static_cast<T>(RAND_MAX);
  }
  FT_Memcpy(gpu_data, cpu_data.get(), size * sizeof(T), MemcpyFlag::kCPU2GPU);
}

TEST_CASE("TensorTest3", "GPU init") {
  easy_transformers::core::Tensor test_tensor(
      easy_transformers::core::NewDLPackTensorT<float>({3, 4}, kDLGPU, 0));
  Fill<float>(test_tensor);
  test_tensor.Print<float>(std::cout);
  REQUIRE(test_tensor.n_dim() == 2);
  REQUIRE(test_tensor.numel() == 3 * 4);
}
#endif

}  // namespace core
}  // namespace easy_transformers
