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

#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif
#include "catch2/catch.hpp"

namespace turbo_transformers {
namespace core {

#ifdef TT_WITH_CUDA
TEST_CASE("CUDADeviceContext", "init") {
  CUDADeviceContext& cuda_ctx = CUDADeviceContext::GetInstance();
  cuda_ctx.Wait();
}

#endif

}  // namespace core
}  // namespace turbo_transformers
