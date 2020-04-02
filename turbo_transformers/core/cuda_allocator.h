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
#include <memory.h>

#include <map>
#include <memory>

#include "macros.h"

namespace turbo_transformers {
namespace core {

class CUDAAllocator {
 public:
  ~CUDAAllocator();

  static CUDAAllocator &GetInstance() {
    static CUDAAllocator instance;
    return instance;
  }

  void *allocate(size_t size);

  void free(void *memory);

 private:
  CUDAAllocator();

  struct AllocatorImpl;
  std::unique_ptr<AllocatorImpl> allocator_;

  DISABLE_COPY_AND_ASSIGN(CUDAAllocator);
};

}  // namespace core
}  // namespace turbo_transformers
