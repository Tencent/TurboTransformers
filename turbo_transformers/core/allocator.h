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

#include <iostream>

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

class StaticAllocator {
 public:
  ~StaticAllocator();

  static StaticAllocator &GetInstance() {
    static StaticAllocator instance;
    return instance;
  }

  void *allocate(std::string name);
  void reserve(int64_t size);
  void schedule(std::unordered_map<std::string, int64_t> *offset_dict) {
    // if (offset_dict_ != nullptr) {
    //   delete offset_dict_;
    // }
    // offset_dict_ = new std::unordered_map<std::string, int64_t>();
    // offset_dict_->insert(offset_dict->begin(), offset_dict->end());
    // offset_dict_ = std::make_shared<std::unordered_map<std::string,
    // int64_t>>(offset_dict);
    // for debug
    // deep copy
    offset_dict_->clear();
    offset_dict_->insert(offset_dict->begin(), offset_dict->end());
    show_offset_dict();
  }
  void show_offset_dict() const {
    std::cerr << "begin show offset dict" << std::endl;
    for (auto it = offset_dict_->begin(); it != offset_dict_->end(); ++it) {
      std::cerr << it->first << ", " << it->second << std::endl;
    }
    std::cerr << "end show offset dict" << std::endl;
  }

 private:
  StaticAllocator();
  void *buff_;
  // std::unordered_map<std::string, int64_t>* offset_dict_;
  std::unique_ptr<std::unordered_map<std::string, int64_t>> offset_dict_;
};

extern void reserve_api(int64_t size);

extern void schedule_api(std::unordered_map<std::string, int64_t> &offset_dict);

}  // namespace core
}  // namespace turbo_transformers
