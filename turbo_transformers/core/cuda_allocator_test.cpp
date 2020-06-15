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
#include "turbo_transformers/core/cuda_allocator.h"

#include <vector>

#include "catch2/catch.hpp"
#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace core {

#ifdef TT_WITH_CUDA
TEST_CASE("cuda_allocator2", "default allocator for tensor") {
  std::vector<size_t> size_list{100, 100, 1000, 256, 200};
  std::vector<void *> addr_list(4);
  for (size_t i = 0; i < size_list.size(); ++i) {
    turbo_transformers::core::Tensor test_tensor(
        turbo_transformers::core::NewDLPackTensorT<float>({size_list[i]},
                                                          kDLGPU));
  }
}

TEST_CASE("cuda_allocator1", "cub") {
  CubCUDAAllocator &cuda_allocator = CubCUDAAllocator::GetInstance();
  std::vector<size_t> size_list{100, 1000, 256, 200};
  std::vector<void *> addr_list(4);
  for (size_t i = 0; i < size_list.size(); ++i) {
    addr_list[i] = cuda_allocator.allocate(size_list[i]);
  }
  for (size_t i = 0; i < size_list.size(); ++i) {
    cuda_allocator.free(addr_list[i]);
  }
}

#endif

}  // namespace core
}  // namespace turbo_transformers
