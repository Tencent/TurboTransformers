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

#pragma once

#include <vector>
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/core/tensor.h"

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
static inline void Copy(const core::Tensor &src, core::Tensor &dst) {
  TT_ENFORCE_EQ(dst.numel(), src.numel(),
                "Copy two tensors should have the same size");
  auto flag = core::ToMemcpyFlag(dst.device_type(), src.device_type());
  core::Memcpy(dst.mutableData<T>(), src.data<T>(), sizeof(T) * src.numel(),
               flag);
}

}  // namespace core
}  // namespace turbo_transformers
