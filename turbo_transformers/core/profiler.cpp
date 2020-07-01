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

#ifdef WITH_PERFTOOLS
#include <chrono>
#include <iostream>
#include <stack>
#include <unordered_map>
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif
#endif

namespace turbo_transformers {
namespace core {
#ifdef WITH_PERFTOOLS
static bool gProfileEnabled = false;

static bool comp(std::pair<std::string, double> a,
                 std::pair<std::string, double> b) {
  return a.second < b.second;
}

struct Profiler::ProfilerImpl {
  void start_profile(const std::string& ctx_name, DLDeviceType dev_type) {
    if (kDLGPU == dev_type) {
#ifdef TT_WITH_CUDA
      cudaEvent_t start_event;
      static auto stream = core::CUDADeviceContext::GetInstance().stream();
      cudaEventCreate(&start_event);
      cudaEventRecord(start_event, stream);
      event_stack_.push(start_event);
#endif
    } else if (kDLCPU == dev_type) {
      auto start = std::chrono::system_clock::now();
      clock_stack_.push(start);
    }
  }
  void end_profile(const std::string& ctx_name, DLDeviceType dev_type) {
    float elapsed_time;
    if (kDLGPU == dev_type) {
#ifdef TT_WITH_CUDA
      cudaEvent_t stop_event;
      cudaEventCreate(&stop_event);
      static auto stream = core::CUDADeviceContext::GetInstance().stream();
      cudaEventRecord(stop_event, stream);
      cudaEventSynchronize(stop_event);
      auto start_event = event_stack_.top();
      event_stack_.pop();
      cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
#endif
    } else if (kDLCPU == dev_type) {
      auto end = std::chrono::system_clock::now();
      if (clock_stack_.empty())
        TT_THROW("Profiler %s has no start time", ctx_name.c_str());
      auto start = clock_stack_.top();
      clock_stack_.pop();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      elapsed_time = float(duration.count()) *
                     std::chrono::microseconds::period::num /
                     std::chrono::microseconds::period::den;
    }

    if (timer_map_.find(ctx_name) != timer_map_.end()) {
      timer_map_[ctx_name] += elapsed_time;
    } else {
      timer_map_.insert({ctx_name, elapsed_time});
    }
  }
  void print_results() const {
    std::cerr << std::endl << profile_name_ << " Time line: " << std::endl;
    std::vector<std::pair<std::string, double>> elems(timer_map_.begin(),
                                                      timer_map_.end());
    std::sort(elems.begin(), elems.end(), comp);
    float total_elapsed = 0.;
    for (auto it = timer_map_.begin(); it != timer_map_.end(); ++it) {
      total_elapsed += it->second;
    }
    for (auto it = elems.begin(); it != elems.end(); ++it) {
      std::cerr << it->first << " , " << it->second << ", "
                << it->second / total_elapsed * 100 << " % " << std::endl;
    }
  }
  void clear() {
    timer_map_.clear();
    while (!clock_stack_.empty()) {
      clock_stack_.pop();
    }

#ifdef TT_WITH_CUDA
    while (!event_stack_.empty()) {
      event_stack_.pop();
    }
#endif
  }
  void set_name(const std::string& profile_name) {
    profile_name_ = profile_name;
  }

 private:
  std::unordered_map<std::string, double> timer_map_;
  std::stack<std::chrono::time_point<std::chrono::system_clock>> clock_stack_;
#ifdef TT_WITH_CUDA
  std::stack<cudaEvent_t> event_stack_;
#endif
  std::string profile_name_;
};

void Profiler::start_profile(const std::string& ctx_name,
                             DLDeviceType dev_type) {
  if (gProfileEnabled) profiler_->start_profile(ctx_name, dev_type);
}

void Profiler::end_profile(const std::string& ctx_name, DLDeviceType dev_type) {
  if (gProfileEnabled) profiler_->end_profile(ctx_name, dev_type);
}

void Profiler::print_results() const {
  if (gProfileEnabled) {
    profiler_->print_results();
  }
}

void Profiler::clear() { profiler_->clear(); }

void Profiler::enable(const std::string& profile_name) {
  gProfileEnabled = true;
  profiler_->set_name(profile_name);
}
void Profiler::disable() { gProfileEnabled = false; }

Profiler::~Profiler() = default;
Profiler::Profiler() : profiler_(new ProfilerImpl()) {}

#endif
void EnableGperf(const std::string& profile_name) {
#ifdef WITH_PERFTOOLS
  LOG_S(1) << "gperf tools enabled. " << profile_name;
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.clear();
  profile_ctx.enable(profile_name);
#else
  LOG_S(WARNING) << "turbo_transformers is not compiled with gperftools.";
#endif
}

void DisableGperf() {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.print_results();
  profile_ctx.disable();
#else
  LOG_S(WARNING) << "turbo_transformers is not compiled with gperftools.";
#endif
}

}  // namespace core
}  // namespace turbo_transformers
