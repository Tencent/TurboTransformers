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

#include "turbo_transformers/core/allocator.h"
#include <unordered_map>

#ifdef TT_WITH_CUDA
#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/core/cuda_enforce.cuh"
#endif

namespace turbo_transformers {
namespace core {

struct BadAlloc : public std::exception {
  explicit BadAlloc(std::string err_msg) : err_str_(err_msg) {}

  const char *what() const noexcept override { return err_str_.c_str(); }

  std::string err_str_;
};

#ifdef TT_WITH_CUDA
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
#endif

namespace {
void *allocate_impl(size_t size, DLDeviceType dev) {
  if (kDLCPU == dev) {
    return align_alloc(size);
  } else if (kDLGPU == dev) {
#ifdef TT_WITH_GPU
    return cuda_alloc(size);
#endif
  } else {
    TT_THROW("Not supported devtype");
  }
  return nullptr;
}

void free_impl(void *memory_addr, DLDeviceType dev) {
  if (kDLCPU == dev) {
    free(memory_addr);
  } else if (kDLGPU == dev) {
#ifdef TT_WITH_GPU
    cuda_free(memory_addr);
#endif
  } else {
    TT_THROW("Not supported devtype");
  }
}

}  // namespace

struct Allocator::BestFitAllocatorImpl {
 public:
  void free_cache(size_t size, DLDeviceType dev) {
    if (size == 0) return;
    size_t cur = 0;
    while (!allocations_.empty()) {  // free the largest
      auto it = --allocations_.end();
      cur += it->first;
      free_impl(it->second, dev);
      addr_size_map_.erase(it->second);
      allocation_size_ -= it->first;
      allocations_.erase(it);
      if (cur >= size) return;
    }
  }

  void *alloc(size_t size, DLDeviceType dev) {
    auto it = allocations_.lower_bound(size);
    void *allocated_addr;
    if (it != allocations_.end() && it->first >= size) {
      allocated_addr = it->second;
      allocations_.erase(it);
    } else {
      try {
        allocated_addr = allocate_impl(size, dev);
      } catch (BadAlloc &) {
        free_cache(size, dev);
        allocated_addr = allocate_impl(size, dev);
      }
    }

    addr_size_map_[allocated_addr] = size;
    return allocated_addr;
  }

  void free(void *data, DLDeviceType dev) {
    auto size = addr_size_map_[data];
    allocations_.emplace(size, data);
    allocation_size_ += size;
    addr_size_map_.erase(data);
  }

  BestFitAllocatorImpl() : allocation_size_(0) {}

 private:
  std::multimap<size_t, void *> allocations_;
  std::unordered_map<void *, size_t> addr_size_map_;
  size_t allocation_size_;
};  // struct Allocator::BestFitAllocatorImpl

struct Allocator::CachingAllocatorImpl {
  void *alloc(size_t size, DLDeviceType dev) {
    void *data = nullptr;
    if (dev == kDLCPU) {
      TT_THROW("Allocator only supports");
    } else if (dev == kDLGPU) {
#ifdef TT_WITH_CUDA
      static auto stream = core::CUDADeviceContext::GetInstance().stream();

      cudaError_t result = cub_allocator.DeviceAllocate(&data, size, stream);
      if (result != cudaSuccess) {
        throw BadAlloc("DeviceAllocate failed.");
      }
#endif
    }
    return data;

  }  // alloc

  void free(void *data, DLDeviceType dev) {
    if (dev == kDLCPU) {
      TT_THROW("Allocator only supports");
    } else if (dev == kDLGPU) {
#ifdef TT_WITH_CUDA
      try {
        cudaError_t result = cub_allocator.DeviceFree(data);
        if (result != cudaErrorCudartUnloading && result != cudaSuccess) {
          throw std::runtime_error("DeviceFree failed ");
        }
      } catch (...) {
      }
#endif
    }
  }

 private:
#ifdef TT_WITH_CUDA
  cub::CachingDeviceAllocator cub_allocator;
#endif
};  // struct Allocator::CachingAllocatorImpl

/*********
 * APIs of Allocator
 *********/
Allocator::Allocator()
    : bestfit_allocator_(new BestFitAllocatorImpl()),
      caching_allocator_(new CachingAllocatorImpl()) {}
Allocator::~Allocator() = default;

void *Allocator::allocate(size_t size, const std::string &strategy,
                          DLDeviceType dev) {
  if ("bestfit" == strategy) {
    return bestfit_allocator_->alloc(size, dev);
  } else if ("cub" == strategy) {
    return caching_allocator_->alloc(size, dev);
  }
}

void Allocator::free(void *memory, const std::string &strategy,
                     DLDeviceType dev) {
  if ("bestfit" == strategy) {
    bestfit_allocator_->free(memory, dev);
  } else if ("cub" == strategy) {
    caching_allocator_->free(memory, dev);
  }
}

}  // namespace core
}  // namespace turbo_transformers
