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
#include "gperftools/profiler.h"
#endif

namespace turbo_transformers {
namespace core {
#ifdef WITH_GPERFTOOLS
static bool gProfileStarted = false;
#endif

void EnableGperf(const std::string &profile_file) {
#ifdef WITH_GPERFTOOLS
  LOG_S(1) << "gperf tools enabled." << profile_file;
  TT_ENFORCE_EQ(gProfileStarted, false, "Currently the gPerf is enabled.");
  ProfilerStart(profile_file.c_str());
  gProfileStarted = true;
#else
  LOG_S(WARNING) << "turbo_transformers is not compiled with gperftools.";
#endif
}

void DisableGperf() {
#ifdef WITH_GPERFTOOLS
  TT_ENFORCE_EQ(gProfileStarted, true, "Currently the gPerf is disabled.");
  ProfilerStop();
  gProfileStarted = false;
#else
  LOG_S(WARNING) << "turbo_transformers is not compiled with gperftools.";
#endif
}

}  // namespace core
}  // namespace turbo_transformers
