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

#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "macros.h"
#include "turbo_transformers/core/allocator_impl.h"

namespace turbo_transformers {
namespace core {
/*
Static Allocator for variable length inputs
 */

class StaticAllocator {
 public:
  ~StaticAllocator();

  static StaticAllocator &GetInstance() {
    static StaticAllocator instance;
    return instance;
  }

  void *allocate(std::string name, DLDeviceType dev = kDLCPU);
  void reserve(int64_t size, DLDeviceType dev = kDLCPU);
  void schedule(std::unordered_map<std::string, int64_t> *offset_dict) {
    // deep copy
    offset_dict_->clear();
    offset_dict_->insert(offset_dict->begin(), offset_dict->end());
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
  void *gpu_buff_;
  std::unique_ptr<std::unordered_map<std::string, int64_t>> offset_dict_;
};

extern void reserve_api(int64_t size, bool use_gpu);

extern void static_schedule_api(
    std::unordered_map<std::string, int64_t> &offset_dict);

/*
Dynamic Allocator for variable length inputs
 */

class DynamicAllocator {
 public:
  ~DynamicAllocator();

  static DynamicAllocator &GetInstance() {
    static DynamicAllocator instance;
    return instance;
  }
  bool isCached(const std::string &name) const {
    auto it = assigned_trunk_->find(name);
    if (it != assigned_trunk_->end() and isOpen()) {
      return true;
    } else {
      return false;
    }
  }

  bool isOpen() const { return is_open_; }
  void Open() { is_open_ = true; }
  void Off() { is_open_ = false; }
  void *allocate(std::string name, DLDeviceType dev);
  void schedule(const std::unordered_map<std::string, int64_t> &assigned_offset,
                const std::unordered_map<std::string, int64_t> &assigned_trunk,
                const std::vector<int64_t> trunk_info,
                const std::string &dev_str);

  void show_offset_dict() const {
    std::cerr << "Lets see assigned_offset" << std::endl;
    for (auto offset_it = assigned_offset_->begin();
         offset_it != assigned_offset_->end(); ++offset_it) {
      auto trunk_it = assigned_trunk_->find(offset_it->first);
      std::cerr << offset_it->first << ", " << trunk_it->second << ", "
                << offset_it->second << std::endl;
    }

    std::cerr << "Lets see trunk_info_" << std::endl;
    for (auto it = trunk_info_->begin(); it != trunk_info_->end(); ++it) {
      std::cerr << *it << std::endl;
    }
  }

 private:
  bool is_open_;
  DynamicAllocator();
  std::vector<void *> gpu_buff_list_;
  std::vector<void *> cpu_buff_list_;
  std::vector<size_t> gpu_mem_size_;
  std::vector<size_t> cpu_mem_size_;

  std::unique_ptr<std::vector<int64_t>> trunk_info_;
  std::unique_ptr<std::unordered_map<std::string, int64_t>> assigned_offset_;
  std::unique_ptr<std::unordered_map<std::string, int64_t>> assigned_trunk_;
};

extern void schedule_dynamic_api(
    const std::unordered_map<std::string, int64_t> &assigned_offset,
    const std::unordered_map<std::string, int64_t> &assigned_trunk,
    const std::vector<int64_t> &trunk_info, const std::string &dev_str);

}  // namespace core
}  // namespace turbo_transformers
