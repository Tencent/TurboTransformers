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
#include "turbo_transformers/core/memory.h"
#ifdef TT_WITH_CUDA
#include <cuda_runtime.h>

#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/core/cuda_enforce.cuh"
#endif
namespace turbo_transformers {
namespace core {
namespace allocator {

struct BadAlloc : public std::exception {
  explicit BadAlloc(std::string err_msg) : err_str_(err_msg) {}

  const char *what() const noexcept override { return err_str_.c_str(); }

  std::string err_str_;
};
extern void *allocate_impl(size_t size, DLDeviceType dev);

extern void free_impl(void *memory_addr, DLDeviceType dev);

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
