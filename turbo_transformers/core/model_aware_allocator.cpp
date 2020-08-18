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

#include "turbo_transformers/core/model_aware_allocator.h"

namespace turbo_transformers {
namespace core {

void *StaticAllocator::allocate(std::string name, DLDeviceType dev) {
  auto it = offset_dict_->find(name);
  if (it != offset_dict_->end()) {
    auto offset = it->second;
    if (dev == kDLCPU) {
      return static_cast<void *>(static_cast<uint8_t *>(buff_) + offset);
    } else {
      return static_cast<void *>(static_cast<uint8_t *>(gpu_buff_) + offset);
    }
  } else {
    TT_THROW("allocate %s failed", name.c_str());
  }
}

void StaticAllocator::reserve(int64_t size, DLDeviceType dev) {
  if (dev == kDLCPU) {
    if (buff_ != nullptr) {
      free_impl(buff_, dev);
    }
    buff_ = static_cast<void *>(allocate_impl(size, dev));
  } else if (dev == kDLGPU) {
    if (gpu_buff_ != nullptr) {
      free_impl(gpu_buff_, dev);
    }
    gpu_buff_ = static_cast<void *>(allocate_impl(size, dev));
  } else {
    TT_THROW("reserve failed %d", dev);
  }
}

StaticAllocator::StaticAllocator()
    : offset_dict_(new std::unordered_map<std::string, int64_t>()) {}

StaticAllocator::~StaticAllocator() = default;

void reserve_api(int64_t size, bool use_gpu) {
  auto &static_allocator = StaticAllocator::GetInstance();
  DLDeviceType dev = kDLCPU;
  if (use_gpu) {
    dev = kDLGPU;
  }
  static_allocator.reserve(static_cast<int64_t>(size), dev);
}

void schedule_api(std::unordered_map<std::string, int64_t> &offset_dict) {
  auto &static_allocator = StaticAllocator::GetInstance();
  static_allocator.schedule(&offset_dict);
}

/*
Dynamic Allocator for variable length inputs
 */
void schedule_dynamic_api(
    const std::unordered_map<std::string, int64_t> &assigned_offset,
    const std::unordered_map<std::string, int64_t> &assigned_trunk,
    const std::vector<int64_t> &trunk_info) {
  auto &dynamica_allocator = DynamicAllocator::GetInstance();
  dynamica_allocator.schedule(assigned_offset, assigned_trunk, trunk_info);
}

// The following coe is time consumming
void DynamicAllocator::schedule(
    const std::unordered_map<std::string, int64_t> &assigned_offset,
    const std::unordered_map<std::string, int64_t> &assigned_trunk,
    const std::vector<int64_t> trunk_info) {
  // deep copy schedule plan
  trunk_info_->clear();
  assigned_offset_->clear();
  assigned_trunk_->clear();

  assigned_offset_->insert(assigned_offset.begin(), assigned_offset.end());
  assigned_trunk_->insert(assigned_trunk.begin(), assigned_trunk.end());
  trunk_info_->assign(trunk_info.begin(), trunk_info.end());

  int Ntrunk = trunk_info.size();
  int Nbuff = gpu_buff_list_.size();
  // update existing trunk, which may be smaller than request
  // because you allocate new trunks and delete old trunks
  for (int i = 0; i < Nbuff && i < Ntrunk; ++i) {
    if (gpu_mem_size_[i] < trunk_info[i]) {
      free_impl(gpu_buff_list_[i], kDLGPU);
      gpu_buff_list_[i] =
          allocate_impl(static_cast<size_t>(trunk_info_->at(i)), kDLGPU);
      gpu_mem_size_[i] = static_cast<size_t>(trunk_info_->at(i));
    }
  }
  // reallocate memory
  for (int i = Nbuff; i < Ntrunk; ++i) {
    gpu_buff_list_.push_back(
        allocate_impl(static_cast<size_t>(trunk_info_->at(i)), kDLGPU));
    gpu_mem_size_.push_back(trunk_info_->at(i));
  }
  for (int i = Nbuff - 1; i >= Ntrunk; --i) {
    free_impl(gpu_buff_list_[i], kDLGPU);
    gpu_buff_list_.pop_back();
    gpu_mem_size_.pop_back();
  }
}

void *DynamicAllocator::allocate(std::string name, DLDeviceType dev) {
  auto offset_it = assigned_offset_->find(name);
  if (offset_it != assigned_offset_->end()) {
    auto offset = offset_it->second;
    auto trunk_id_it = assigned_trunk_->find(name);
    if (trunk_id_it != assigned_trunk_->end()) {
      auto trunk_id = trunk_id_it->second;
      if (dev == kDLGPU) {
        return static_cast<void *>(
            static_cast<uint8_t *>(gpu_buff_list_[trunk_id]) + offset);
      } else {
        TT_THROW("DynamicAllocator allocator dose not support CPU.");
      }
    }
  } else {
    TT_THROW("allocate %s failed", name.c_str());
  }
}

DynamicAllocator::DynamicAllocator()
    : trunk_info_(new std::vector<int64_t>()),
      assigned_offset_(new std::unordered_map<std::string, int64_t>()),
      assigned_trunk_(new std::unordered_map<std::string, int64_t>()) {}

DynamicAllocator::~DynamicAllocator() = default;

}  // namespace core
}  // namespace turbo_transformers
