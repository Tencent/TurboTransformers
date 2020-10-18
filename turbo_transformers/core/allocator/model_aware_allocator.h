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
#include <set>

#include "turbo_transformers/core/allocator/allocator_impl.h"
#include "turbo_transformers/core/allocator/base_allocator.h"
#include "turbo_transformers/core/allocator/bert_config.h"
#include "turbo_transformers/core/allocator/model_aware_memory_scheduler.h"

namespace turbo_transformers {
namespace core {
namespace allocator {

/***
 * Model Aware Allocator,
 * which allocate memory for all activations after the parameters are
 * determined.
 * TODO(jiaruifang) Add support for GPU
 */
class ModelAwareAllocator : public BaseAllocator {
 public:
  explicit ModelAwareAllocator(const std::string& model_name)
      : model_name_(model_name),
#ifdef TT_WITH_CUDA
        gpu_chunk_list_(
            [&](size_t size) -> char* {
              return (char*)allocate_impl(size, kDLGPU);
            },
            [&](void* mem_addr) { free_impl(mem_addr, kDLGPU); }),
#endif
        cpu_chunk_list_(
            [&](size_t size) -> char* {
              return (char*)allocate_impl(size, kDLCPU);
            },
            [&](void* mem_addr) { free_impl(mem_addr, kDLCPU); }) {
    if (model_name == "bert") {
    } else {
      TT_THROW("ModelAwareAllocator dose not support %s", model_name.c_str());
    }
  }
  bool is_activation(const std::string& name) const override {
    return activation_names_.count(name) != 0;
  }
  // reset memory schema, after input tensor changes.
  void reset(std::vector<int64_t>& param_list) override {
    int64_t batch_size = param_list[0];
    int64_t seq_len = param_list[1];
    int64_t num_head = param_list[2];
    int64_t hidden_size = param_list[3];
    int64_t num_layer = param_list[4];

    bert_config::GetBertTensorUsageRecord<float>(
        tensor_usage_records_, activation_names_, batch_size, seq_len, num_head,
        hidden_size, num_layer);
    ChunkedGreedyBySizeOffsetCalculation(tensor_usage_records_, cpu_chunk_list_,
                                         cpu_tensor_position_map_);
#ifdef TT_WITH_CUDA
    ChunkedGreedyBySizeOffsetCalculation(tensor_usage_records_, gpu_chunk_list_,
                                         gpu_tensor_position_map_);
#endif
  }

  // return memory address according to the tensor name
  // if the tensor name is "", indicating the memory is not the activation of
  // DNN model, allocate a piece of memory on heap.
  void* allocate(size_t size, DLDeviceType dev,
                 const std::string& name) override {
    if (!is_activation(name)) {
      void* addr = allocate_impl(size, dev);
      return addr;
    }
    if (kDLCPU == dev) {
      auto it = cpu_tensor_position_map_.find(name);
      TT_ENFORCE(it != cpu_tensor_position_map_.end(),
                 "ModelAwareAllocator allocate %s failed", name.c_str());
      Chunk* chunk_ptr = it->second.chunk_ptr_;
      return static_cast<void*>(static_cast<uint8_t*>(chunk_ptr->GetMemAddr()) +
                                it->second.offset_);
    } else {
#ifdef TT_WITH_CUDA
      auto it = gpu_tensor_position_map_.find(name);
      TT_ENFORCE(it != gpu_tensor_position_map_.end(),
                 "ModelAwareAllocator allocate %s failed", name.c_str());
      Chunk* chunk_ptr = it->second.chunk_ptr_;
      return static_cast<void*>(static_cast<uint8_t*>(chunk_ptr->GetMemAddr()) +
                                it->second.offset_);
#endif
    }
  }

  void free(void* mem, DLDeviceType dev, const std::string& name) override {
    if (!is_activation(name)) {
      return free_impl(mem, dev);
    }
  }
  ~ModelAwareAllocator();

 private:
  std::string model_name_;
  std::vector<TensorRecordItemPtr> tensor_usage_records_;
  std::map<std::string, TensorPositionInfo> cpu_tensor_position_map_;
  ChunkList cpu_chunk_list_;
  std::set<std::string> activation_names_;
#ifdef TT_WITH_CUDA
  std::map<std::string, TensorPositionInfo> gpu_tensor_position_map_;
  ChunkList gpu_chunk_list_;
#endif
};

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
