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
#include "turbo_transformers/core/allocator/allocator_impl.h"
#include "turbo_transformers/core/allocator/base_allocator.h"

#ifdef TT_WITH_CUDA
#include <cuda_runtime.h>

#include <cub/util_allocator.cuh>

#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/core/cuda_enforce.cuh"
#endif

namespace turbo_transformers {
namespace core {
namespace allocator {

/**
 * CPU allocates memory on heap..
 * GPU uses nvidia cub to manage gpu memory
 * TODO(jiaruifang) polish this code.
 */
class NaiveAllocator : public BaseAllocator {
 public:
#ifdef TT_WITH_CUDA
  NaiveAllocator() : cub_allocator(unsigned(8)) {}
#else
#endif
  void* allocate(size_t size, DLDeviceType dev,
                 const std::string& name) override {
    void* data = nullptr;
    if (dev == kDLCPU) {
      return allocate_impl(size, kDLCPU);
    } else if (dev == kDLGPU) {
#ifdef TT_WITH_CUDA
      static auto stream = core::CUDADeviceContext::GetInstance().stream();
      try {
        cudaError_t result = cub_allocator.DeviceAllocate(&data, size, stream);
        if (result != cudaSuccess) {
          throw BadAlloc("DeviceAllocate failed.");
        }
      } catch (...) {
        cub_allocator.FreeAllCached();
        cudaError_t result = cub_allocator.DeviceAllocate(&data, size, stream);
        if (result != cudaSuccess) {
          std::stringstream ss;
          ss << "DeviceAllocate failed Again. " << size;
          throw BadAlloc(ss.str());
        }
      }
#endif
    }
    return data;
  }  // alloc

  void free(void* mem, DLDeviceType dev, const std::string& name) override {
    if (dev == kDLCPU) {
      free_impl(mem, kDLCPU);
    } else if (dev == kDLGPU) {
#ifdef TT_WITH_CUDA
      try {
        cudaError_t result = cub_allocator.DeviceFree(mem);
        if (result != cudaErrorCudartUnloading && result != cudaSuccess) {
          throw std::runtime_error("DeviceFree failed ");
        }
      } catch (...) {
      }
#endif
    }
  }

  ~NaiveAllocator();

 private:
#ifdef TT_WITH_CUDA
  cub::CachingDeviceAllocator cub_allocator;
  /*
  (  unsigned int   bin_growth,
    unsigned int   min_bin = 1,
    unsigned int   max_bin = INVALID_BIN,
    size_t   max_cached_bytes = INVALID_SIZE,
    bool   skip_cleanup = false,
    bool   debug = false
  )
   */
#endif
};

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
