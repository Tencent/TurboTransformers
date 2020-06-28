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

#ifdef WITH_PERFTOOLS
#include <dlpack/dlpack.h>

#include <memory>

#include "macros.h"
#endif

namespace turbo_transformers {
namespace core {

#ifdef WITH_PERFTOOLS
class Profiler {
 public:
  ~Profiler();
  static Profiler& GetInstance() {
    static Profiler instance;
    return instance;
  }
  void clear();
  void start_profile(const std::string& ctx_name,
                     DLDeviceType dev_type = kDLCPU);
  void end_profile(const std::string& ctx_name, DLDeviceType dev_type = kDLCPU);
  void print_results() const;
  void enable(const std::string& profile_name);
  void disable();

 private:
  Profiler();

  struct ProfilerImpl;
  std::unique_ptr<ProfilerImpl> profiler_;

  DISABLE_COPY_AND_ASSIGN(Profiler);
};
#endif
void EnableGperf(const std::string& profile_file);
void DisableGperf();

}  // namespace core
}  // namespace turbo_transformers
