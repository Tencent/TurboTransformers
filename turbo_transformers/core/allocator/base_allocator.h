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
#include <string>
#include <vector>

#include "dlpack/dlpack.h"

namespace turbo_transformers {
namespace core {
namespace allocator {

class BaseAllocator {
 public:
  virtual void* allocate(size_t size, DLDeviceType dev,
                         const std::string& name) = 0;
  virtual void free(void* mem, DLDeviceType dev, const std::string& name) = 0;
  // an interface to modify model-aware allocator's model config.
  // the config is encoded in a list of int64_t
  virtual void reset(std::vector<int64_t>& configs){};
  virtual bool is_activation(const std::string& name) const { return false; };
  // TODO(jiaruifang) release all memory cached.
  virtual void release(){};
  virtual ~BaseAllocator();
};

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
