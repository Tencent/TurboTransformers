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
#include <unordered_map>
#include "macros.h"
#include "turbo_transformers/core/memory.h"

namespace turbo_transformers {
namespace core {

class Allocator {
 public:
  ~Allocator();

  static Allocator &GetInstance() {
    static Allocator instance;
    return instance;
  }

  void *allocate(size_t size, const std::string &strategy, DLDeviceType dev);

  void free(void *memory, const std::string &strategy, DLDeviceType dev);

 private:
  Allocator();
  struct BestFitAllocatorImpl;
  std::unique_ptr<BestFitAllocatorImpl> bestfit_allocator_;
  struct CachingAllocatorImpl;
  std::unique_ptr<CachingAllocatorImpl> caching_allocator_;

  DISABLE_COPY_AND_ASSIGN(Allocator);
};

}  // namespace core
}  // namespace turbo_transformers
