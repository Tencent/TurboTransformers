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

#include "profiler.h"

#include "enforce.h"
#include "loguru.hpp"

#ifdef WITH_GPERFTOOLS
#include <chrono>
#include <iostream>
#include <stack>
#include <unordered_map>
#endif

namespace turbo_transformers {
namespace core {
#ifdef WITH_GPERFTOOLS
static bool gProfileStarted = false;

struct Profiler::ProfilerImpl {
  void start_profile(const std::string& ctx_name) {
    auto start = std::chrono::system_clock::now();
    clock_stack_.push(start);
  }
  void end_profile(const std::string& ctx_name) {
    auto end = std::chrono::system_clock::now();
    if (clock_stack_.empty()) TT_THROW("Profiler has no start time");
    auto start = clock_stack_.top();
    clock_stack_.pop();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto elapsed_time = double(duration.count()) *
                        std::chrono::microseconds::period::num /
                        std::chrono::microseconds::period::den;
    if (timer_map_.find(ctx_name) != timer_map_.end()) {
      timer_map_[ctx_name] += elapsed_time;
    } else {
      timer_map_.insert({ctx_name, elapsed_time});
    }
  }
  void print_results() const {
    std::cerr << "Time line in print_results " << std::endl;
    for (auto it = timer_map_.begin(); it != timer_map_.end(); ++it) {
      std::cerr << it->first << " , " << it->second << std::endl;
    }
  }
  void clear() {
    timer_map_.clear();
    while (!clock_stack_.empty()) {
      clock_stack_.pop();
    }
  }

 private:
  std::unordered_map<std::string, double> timer_map_;
  std::stack<std::chrono::time_point<std::chrono::system_clock>> clock_stack_;
};

void Profiler::start_profile(const std::string& ctx_name) {
  profiler_->start_profile(ctx_name);
}

void Profiler::end_profile(const std::string& ctx_name) {
  profiler_->end_profile(ctx_name);
}

void Profiler::print_results() const { profiler_->print_results(); }

void Profiler::clear() { profiler_->clear(); }

Profiler::~Profiler() = default;
Profiler::Profiler() : profiler_(new ProfilerImpl()) {}

#endif
void EnableGperf(const std::string& profile_file) {
#ifdef WITH_GPERFTOOLS
  LOG_S(1) << "gperf tools enabled." << profile_file;
  TT_ENFORCE_EQ(gProfileStarted, false, "Currently the gPerf is enabled.");
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.clear();
  gProfileStarted = true;
#else
  LOG_S(WARNING) << "turbo_transformers is not compiled with gperftools.";
#endif
}

void DisableGperf() {
#ifdef WITH_GPERFTOOLS
  TT_ENFORCE_EQ(gProfileStarted, true, "Currently the gPerf is disabled.");
  gProfileStarted = false;
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.print_results();
#else
  LOG_S(WARNING) << "turbo_transformers is not compiled with gperftools.";
#endif
}

}  // namespace core
}  // namespace turbo_transformers
