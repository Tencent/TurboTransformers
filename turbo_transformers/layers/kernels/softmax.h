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

#pragma once
#include <cstdint>

#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {
extern void ApplyMaskAndSoftmax(core::Tensor* inout,
                                const core::Tensor& att_mask, float scale,
                                const std::string name = "ApplyMaskAndSoftmax");

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
