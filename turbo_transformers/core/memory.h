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
#include "dlpack/dlpack.h"
#include "turbo_transformers/core/enforce.h"
namespace turbo_transformers {
namespace core {

extern void *align_alloc(size_t sz, size_t align = 64);

template <typename T>
inline T *align_alloc_t(size_t sz, size_t align = 64) {
  return reinterpret_cast<T *>(align_alloc(sz * sizeof(T), align));
}

enum class MemcpyFlag {
  kCPU2GPU = 0,
  kGPU2CPU,
  kCPU2CPU,
  kGPU2GPU,
  kNUM_MEMCPY_FLAGS
};

extern void Memcpy(void *dst_data, const void *src_data, size_t data_size,
                   MemcpyFlag flag);

extern MemcpyFlag ToMemcpyFlag(DLDeviceType dst, DLDeviceType src);

}  // namespace core
}  // namespace turbo_transformers
