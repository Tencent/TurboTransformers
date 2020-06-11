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
#include <memory.h>

#include <map>
#include <memory>

#include "macros.h"

namespace turbo_transformers {
namespace core {

class CUDAAllocator {
 public:
  virtual ~CUDAAllocator() = default;

  virtual void *allocate(size_t size) = 0;

  virtual void free(void *memory) = 0;

 private:
  CUDAAllocator() = default;

  DISABLE_COPY_AND_ASSIGN(CUDAAllocator);
};

class CubCUDAAllocator : public CUDAAllocator {
 public:
  static CubCUDAAllocator &GetInstance() {
    static CubCUDAAllocator instance;
    return instance;
  }
  virtual void *allocate(size_t size) override;
  virtual void free(void *memory) override;
  ~CubCUDAAllocator = default;

 private:
  CubCUDAAllocator();
  struct CubAllocatorImpl;
  std::unique_ptr<CubAllocatorImpl> allocator_;
}

class BestFitCUDAAllocator : public CUDAAllocator {
 public:
  static BestFitCUDAAllocator &GetInstance() {
    static BestFitCUDAAllocator instance;
    return instance;
  }
  virtual void *allocate(size_t size) override;
  virtual void free(void *memory) override;
  ~BestFitCUDAAllocator = default;

 private:
  BestFitCUDAAllocator();
  struct BestFitAllocatorImpl;
  std::unique_ptr<BestFitAllocatorImpl> allocator_;
}

}  // namespace core
}  // namespace turbo_transformers
