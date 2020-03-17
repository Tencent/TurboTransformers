// Copyright 2020 Tencent
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "turbo_transformers/core/eigen-tensor.h"

#include <thread>
#include <unsupported/Eigen/CXX11/ThreadPool>

namespace turbo_transformers {
namespace core {

static int get_thread_num() {
  char* env = getenv("OMP_NUM_THREADS");
  if (env == nullptr || std::string(env) == "0") {
    return std::thread::hardware_concurrency();
  } else {
    int thnum = std::atoi(env);
    if (thnum <= 0) {
      return std::thread::hardware_concurrency();
    } else {
      return thnum;
    }
  }
}

Eigen::ThreadPoolDevice& CPUDevice() {
  static Eigen::ThreadPool pool(get_thread_num());
  static Eigen::ThreadPoolDevice device(&pool, get_thread_num());
  return device;
}

}  // namespace core
}  // namespace turbo_transformers
