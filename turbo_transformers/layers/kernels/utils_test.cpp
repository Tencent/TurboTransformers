// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.
#include "turbo_transformers/layers/kernels/utils.h"

#include "catch2/catch.hpp"

namespace turbo_transformers {
namespace layers {
namespace kernels {

TEST_CASE("cpu-concate", "test1") {
  turbo_transformers::core::Tensor t1(
      turbo_transformers::core::NewDLPackTensorT<float>({2, 1, 2, 2}));
  turbo_transformers::core::Tensor t2(
      turbo_transformers::core::NewDLPackTensorT<float>({2, 1, 3, 2}));
  for (int i = 0; i < t1.numel(); ++i) {
    t1.mutableData<float>()[i] = i * 1.0;
  }

  for (int i = 0; i < t2.numel(); ++i) {
    t2.mutableData<float>()[i] = i * 100.0;
  }
  turbo_transformers::core::Tensor res1(nullptr), res2(nullptr), res3(nullptr);
  Concat<float>(t1, t2, 2, &res1);
  // res1.Print<float>(std::cerr);

  turbo_transformers::core::Tensor t3(
      turbo_transformers::core::NewDLPackTensorT<float>({4, 2}));
  turbo_transformers::core::Tensor t4(
      turbo_transformers::core::NewDLPackTensorT<float>({3, 2}));
  Concat<float>(t3, t4, 0, &res2);

  turbo_transformers::core::Tensor t5(
      turbo_transformers::core::NewDLPackTensorT<float>({2, 3}));
  turbo_transformers::core::Tensor t6(
      turbo_transformers::core::NewDLPackTensorT<float>({2, 4}));
  for (int i = 0; i < t5.numel(); ++i) {
    t5.mutableData<float>()[i] = i * 1.0;
  }

  for (int i = 0; i < t6.numel(); ++i) {
    t6.mutableData<float>()[i] = i * 100.0;
  }
  Concat<float>(t5, t6, 1, &res2);
  // res3.Print<float>(std::cerr);

  // t5.Print<float>(std::cerr);
  // t6.Print<float>(std::cerr);
  // res3.Print<float>(std::cerr);
  // REQUIRE(test_tensor.n_dim() == 2);
  // REQUIRE(test_tensor.numel() == 3 * 4);
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
