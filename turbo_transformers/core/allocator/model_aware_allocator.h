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

  // reset memory schema, after input tensor changes.
  void reset(std::vector<int64_t>& param_list) override {
    LOG_SCOPE_F(INFO,
                "start schedule memory offsets for model aware allocator.");
    int64_t batch_size = param_list[0];
    int64_t seq_len = param_list[1];
    int64_t num_head = param_list[2];
    int64_t hidden_size = param_list[3];
    int64_t num_layer = param_list[4];

    bert_config::GetBertTensorUsageRecord<float>(tensor_usage_records_,
                                                 batch_size, seq_len, num_head,
                                                 hidden_size, num_layer);
    ChunkedGreedyBySizeOffsetCalculation(tensor_usage_records_, cpu_chunk_list_,
                                         cpu_tensor_position_map_);
  }

  // return memory address according to the tensor name
  // if the tensor name is "", inidicating the memory is not the activation of
  // DNN model, allocate a piece of memory on heap.
  void* allocate(size_t size, DLDeviceType dev,
                 const std::string& name) override {
    if (name.size() == 0) {
      return allocate_impl(size, dev);
    }
    if (kDLCPU == dev) {
      auto it = cpu_tensor_position_map_.find(name);
      TT_ENFORCE(it != cpu_tensor_position_map_.end(),
                 "ModelAwareAllocator allocate %s failed", name.c_str());
      Chunk* chunk_ptr = it->second.chunk_ptr_;
      return static_cast<void*>(static_cast<uint8_t*>(chunk_ptr->GetMemAddr()) +
                                it->second.offset_);
    } else {
      TT_THROW("Model Aware Allocator has not been implemented!");
    }
  }

  void free(void* mem, DLDeviceType dev, const std::string& name) override {}
  ~ModelAwareAllocator();

 private:
  std::string model_name_;
  std::vector<TensorRecordItemPtr> tensor_usage_records_;
  std::map<std::string, TensorPositionInfo> cpu_tensor_position_map_;
  ChunkList cpu_chunk_list_;
#ifdef TT_WITH_CUDA
  std::map<std::string, TensorPositionInfo> gpu_tensor_position_map_;
  ChunkList gpu_chunk_list_;
#endif
};

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
