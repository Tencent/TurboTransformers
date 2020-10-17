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
#include <dlpack/dlpack.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "turbo_transformers/core/macros.h"

namespace turbo_transformers {
namespace core {
namespace allocator {

/***
 * If the runtime detect the GPU, then init a GPU allocator as well as a CPU
 * one. If no GPU detected, only init a CPU allocator. In this way, we have to
 * pass a device parameter to the allocate and free API. The device type have to
 * be determined when call allocate.
 */
class Allocator {
 public:
  ~Allocator();

  static Allocator& GetInstance() {
    static Allocator instance;
    return instance;
  }

  void set_schema(const std::string& schema);

  std::string get_schema() const;

  void set_config(std::vector<int64_t> configs);
  void* allocate(size_t size, DLDeviceType dev, const std::string& name = "");
  void free(void* memory, DLDeviceType dev, const std::string& name = "");
  bool is_activation(const std::string& name);

 private:
  void register_schema(const std::string& schema);
  Allocator();
  struct AllocatorImpl;
  std::unique_ptr<AllocatorImpl> impl_;

  DISABLE_COPY_AND_ASSIGN(Allocator);
};

extern void bert_opt_mem_allocate_api(int64_t batch_size, int64_t seq_len,
                                      int64_t num_head, int64_t hidden_size,
                                      int64_t num_layer,
                                      const std::string& dev_str);

extern void reset_allocator_schema(const std::string& name);
}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
