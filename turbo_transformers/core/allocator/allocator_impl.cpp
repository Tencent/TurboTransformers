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

#include "allocator_impl.h"

#include "turbo_transformers/core/memory.h"
namespace turbo_transformers {
namespace core {
namespace allocator {
#ifdef TT_WITH_CUDA
static void *cuda_alloc(size_t sz) {
  void *device_mem;
  if (cudaMalloc((void **)&(device_mem), sz) != cudaSuccess) {
    throw BadAlloc("cudaMalloc failed.");
  }
  return device_mem;
}

static void cuda_free(void *data) { TT_ENFORCE_CUDA_SUCCESS(cudaFree(data)); }
#endif

void *allocate_impl(size_t size, DLDeviceType dev) {
  if (kDLCPU == dev) {
    return align_alloc(size);
  } else if (kDLGPU == dev) {
#ifdef TT_WITH_CUDA
    auto addr = cuda_alloc(size);
    return addr;
#endif
  } else {
    TT_THROW("Not supported devtype id as %d", dev);
  }
  return nullptr;
}

void free_impl(void *memory_addr, DLDeviceType dev) {
  if (kDLCPU == dev) {
    free(memory_addr);
  } else if (kDLGPU == dev) {
#ifdef TT_WITH_CUDA
    cuda_free(memory_addr);
#endif
  } else {
    TT_THROW("Not supported devtype id as %d", dev);
  }
}
}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
