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

extern void FT_Memcpy(void *dst_data, const void *src_data, size_t data_size,
                      MemcpyFlag flag);

extern MemcpyFlag ToMemcpyFlag(DLDeviceType dst, DLDeviceType src);

}  // namespace core
}  // namespace turbo_transformers
