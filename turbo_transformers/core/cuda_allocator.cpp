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
#include "turbo_transformers/core/cuda_allocator.h"

#include <cuda_runtime.h>

#include <cub/util_allocator.cuh>

#include "turbo_transformers/core/cuda_device_context.h"

namespace turbo_transformers {
namespace core {

struct BadAlloc : public std::exception {
  explicit BadAlloc(std::string err_msg) : err_str_(err_msg) {}

  const char *what() const noexcept override { return err_str_.c_str(); }

  std::string err_str_;
};

struct CUDAAllocator::AllocatorImpl {
  void *alloc(size_t size) {
    static auto stream = core::CUDADeviceContext::GetInstance().stream();
    void *data = nullptr;
    cudaError_t result = cub_allocator.DeviceAllocate(&data, size, stream);
    if (result != cudaSuccess) {
      throw BadAlloc("DeviceAllocate failed.");
    }
    return data;
  }

  void free(void *data) {
    try {
      cudaError_t result = cub_allocator.DeviceFree(data);
      if (result != cudaErrorCudartUnloading && result != cudaSuccess) {
        throw std::runtime_error("DeviceFree failed ");
      }
    } catch (...) {
    }
  }

  void free_all_cache() { cub_allocator.FreeAllCached(); }

  ~AllocatorImpl() { cub_allocator.FreeAllCached(); }

  cub::CachingDeviceAllocator cub_allocator;
};

CUDAAllocator::CUDAAllocator() : allocator_(new AllocatorImpl()) {}

CUDAAllocator::~CUDAAllocator() = default;

void *CUDAAllocator::allocate(size_t size) {
  try {
    return allocator_->alloc(size);
  } catch (BadAlloc &) {
    allocator_->free_all_cache();
    return allocator_->alloc(size);
  }
}

void CUDAAllocator::free(void *memory) { allocator_->free(memory); }

}  // namespace core
}  // namespace turbo_transformers
