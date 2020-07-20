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

#include <vector>

#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/core/tensor.h"

#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace core {
template <typename T>
static inline void Copy(const T *data, size_t size, DLDeviceType srcDevice,
                        DLDeviceType dstDevice, T *dst) {
  auto flag = core::ToMemcpyFlag(dstDevice, srcDevice);
  core::Memcpy(dst, data, sizeof(T) * size, flag);
}
template <typename T>
static inline void Copy(const T *data, size_t size, DLDeviceType srcDevice,
                        core::Tensor &dst) {
  DLDeviceType dstDevice = dst.device_type();
  auto flag = core::ToMemcpyFlag(dstDevice, srcDevice);
  core::Memcpy(dst.mutableData<T>(), data, sizeof(T) * size, flag);
}
template <typename T>
static inline void Copy(const core::Tensor &src, std::vector<T> &dst) {
  auto flag = core::ToMemcpyFlag(DLDeviceType::kDLCPU, src.device_type());
  core::Memcpy(dst.data(), src.data<T>(), sizeof(T) * src.numel(), flag);
}
template <typename T>
static inline void Copy(const core::Tensor &src, core::Tensor &dst,
                        const std::string name = "Copy") {
#ifdef WITH_PERFTOOLS
  auto &profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, src.device_type());
#endif
  TT_ENFORCE_EQ(dst.numel(), src.numel(),
                "Copy two tensors should have the same size");
  auto flag = core::ToMemcpyFlag(dst.device_type(), src.device_type());
  core::Memcpy(dst.mutableData<T>(), src.data<T>(), sizeof(T) * src.numel(),
               flag);
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, src.device_type());
#endif
}

}  // namespace core
}  // namespace turbo_transformers
