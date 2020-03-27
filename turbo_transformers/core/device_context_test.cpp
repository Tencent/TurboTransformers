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

#ifdef FT_WITH_CUDA
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
