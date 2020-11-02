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

#include "turbo_transformers/core/allocator/allocator_api.h"

#include <map>
#include <memory>

#include "turbo_transformers/core/allocator/allocator_impl.h"
#include "turbo_transformers/core/allocator/base_allocator.h"
#include "turbo_transformers/core/allocator/model_aware_allocator.h"
#include "turbo_transformers/core/allocator/naive_allocator.h"

namespace turbo_transformers {
namespace core {
namespace allocator {

struct Allocator::AllocatorImpl {
  void set_schema(const std::string& schema) { schema_ = schema; }

  void* allocate(size_t size, DLDeviceType dev, const std::string& name) {
    auto it = allocators.find(schema_);
    TT_ENFORCE(it != allocators.end(), "Allocator scheme %s is not valid",
               schema_.c_str());
    return it->second->allocate(size, dev, name);
  }

  void free(void* memory, DLDeviceType dev, const std::string& name) {
    auto it = allocators.find(schema_);
    TT_ENFORCE(it != allocators.end(), "Allocator scheme %s is not valid",
               schema_.c_str());
    it->second->free(memory, dev, name);
  }

  std::string schema_;
  std::map<const std::string, std::unique_ptr<BaseAllocator>> allocators;
};

Allocator::~Allocator() = default;
/**
 * Defualt constructor.
 * Dev : CPU or GPU
 * set a default schema. CPU uses naiveAllocator, GPU uses cubAllocator
 */
Allocator::Allocator() : impl_(new AllocatorImpl()) {
  register_schema("naive");
  register_schema("model-aware");
  impl_->set_schema("naive");
}

void* Allocator::allocate(size_t size, DLDeviceType dev,
                          const std::string& name) {
  return impl_->allocate(size, dev, name);
}

void Allocator::free(void* memory, DLDeviceType dev, const std::string& name) {
  impl_->free(memory, dev, name);
}

std::string Allocator::get_schema() const { return impl_->schema_; };

/**
 * We should register both CPU and GPU allocator
 * @param schema one of {"naive", "model-aware"}
 */
void Allocator::register_schema(const std::string& schema) {
  if ("model-aware" == schema) {
    impl_->allocators.emplace("model-aware", new ModelAwareAllocator("bert"));
  } else if ("naive" == schema) {
    impl_->allocators.emplace("naive", new NaiveAllocator());
  } else {
    TT_THROW("no schem %s in register_schema", schema.c_str());
  }
}

void Allocator::set_schema(const std::string& schema) {
  LOG_S(INFO) << "Global allocator has be set to " << schema;
  impl_->schema_ = schema;
}

void Allocator::set_config(std::vector<int64_t> configs) {
  auto it = impl_->allocators.find("model-aware");
  if (it != impl_->allocators.end()) {
    it->second->reset(configs);
  }
}

/***
 * If use the model-aware model, judge if the name is an activation.
 * @param name : activation name.
 * @return : is the activation a activation.
 */
bool Allocator::is_activation(const std::string& name) {
  if (get_schema() != "model-aware") return false;
  auto it = impl_->allocators.find("model-aware");
  if (it != impl_->allocators.end()) {
    return it->second->is_activation(name);
  }
  return false;
}

void bert_opt_mem_allocate_api(int64_t batch_size, int64_t seq_len,
                               int64_t num_head, int64_t hidden_size,
                               int64_t num_layer, const std::string& dev_str) {
  auto& allocator = Allocator::GetInstance();
  // TODO(jiaruifang) we can only schedule the dev
  allocator.set_config({batch_size, seq_len, num_head, hidden_size, num_layer});
}

extern void reset_allocator_schema(const std::string& name) {
  auto& allocator = Allocator::GetInstance();
  LOG_S(INFO) << "The Global Allocator has been switch from "
              << allocator.get_schema() << " to " << name;
  allocator.set_schema(name);
}

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
