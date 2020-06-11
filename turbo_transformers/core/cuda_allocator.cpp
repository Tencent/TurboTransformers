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
#include "turbo_transformers/core/cuda_allocator.h"
#include <cuda_runtime.h>
#include "turbo_transformers/core/cuda_enforce.cuh"

#include <cub/util_allocator.cuh>
#include <unordered_map>

#include "turbo_transformers/core/cuda_device_context.h"

namespace turbo_transformers {
namespace core {

struct BadAlloc : public std::exception {
  explicit BadAlloc(std::string err_msg) : err_str_(err_msg) {}

  const char *what() const noexcept override { return err_str_.c_str(); }

  std::string err_str_;
};

CUDAAllocator::CUDAAllocator() = default;

/**********
 * Allocator using cub Caching Memory algorithm
 **********/

struct CubCUDAAllocator::CubAllocatorImpl {
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

  ~CubAllocatorImpl() { cub_allocator.FreeAllCached(); }

  cub::CachingDeviceAllocator cub_allocator;
};

CubCUDAAllocator::CubCUDAAllocator() : allocator_(new CubAllocatorImpl()) {}

void *CubCUDAAllocator::allocate(size_t size) {
  try {
    return allocator_->alloc(size);
  } catch (BadAlloc &) {
    allocator_->free_all_cache();
    return allocator_->alloc(size);
  }
}

void CubCUDAAllocator::free(void *memory) { allocator_->free(memory); }
CubCUDAAllocator::~CubCUDAAllocator() = default;

/**********
 * Allocator using best fit algorithm
 **********/

static void *cuda_alloc(size_t sz) {
  void *device_mem;
  try {
    TT_ENFORCE_CUDA_SUCCESS(cudaMalloc((void **)&(device_mem), sz));
  } catch (...) {
    throw BadAlloc("cudaMalloc failed.");
  }
  return device_mem;
}

static void cuda_free(void *data) { TT_ENFORCE_CUDA_SUCCESS(cudaFree(data)); }

struct BestFitCUDAAllocator::BestFitAllocatorImpl {
 public:
  void free_cache(size_t size) {
    if (size == 0) return;
    size_t cur = 0;
    while (!allocations_.empty()) {  // free the largest
      auto it = --allocations_.end();
      cur += it->first;
      cuda_free(it->second);
      addr_size_map_.erase(it->second);
      allocation_size_ -= it->first;
      allocations_.erase(it);
      if (cur >= size) return;
    }
  }

  void *alloc(size_t size) {
    auto it = allocations_.lower_bound(size);
    void *allocated_addr;
    if (it != allocations_.end() && it->first >= size) {
      allocated_addr = it->second;
      allocations_.erase(it);
    }

    try {
      allocated_addr = cuda_alloc(size);
    } catch (BadAlloc &) {
      free_cache(size);
      allocated_addr = cuda_alloc(size);
    }
    addr_size_map_[allocated_addr] = size;
    return allocated_addr;
  }

  void free(void *data) {
    auto size = addr_size_map_[data];
    allocations_.emplace(size, data);
    allocation_size_ += size;
    addr_size_map_.erase(data);
  }

  BestFitAllocatorImpl() : allocation_size_(0) {}
  ~BestFitAllocatorImpl() = default;

 private:
  std::multimap<size_t, void *> allocations_;
  std::unordered_map<void *, size_t> addr_size_map_;
  size_t allocation_size_;
};

BestFitCUDAAllocator::BestFitCUDAAllocator()
    : allocator_(new BestFitAllocatorImpl()) {}

void *BestFitCUDAAllocator::allocate(size_t size) {
  try {
    return allocator_->alloc(size);
  } catch (BadAlloc &) {
    allocator_->free_cache(size);
    return allocator_->alloc(size);
  }
}

void BestFitCUDAAllocator::free(void *memory) { allocator_->free(memory); }

BestFitCUDAAllocator::~BestFitCUDAAllocator() = default;

}  // namespace core
}  // namespace turbo_transformers
