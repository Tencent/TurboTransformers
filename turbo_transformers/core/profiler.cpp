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
  FT_ENFORCE_EQ(gProfileStarted, false, "Currently the gPerf is enabled.");
  ProfilerStart(profile_file.c_str());
  gProfileStarted = true;
#else
  LOG_S(WARNING) << "turbo_transformers is not compiled with gperftools.";
#endif
}

void DisableGperf() {
#ifdef WITH_GPERFTOOLS
  FT_ENFORCE_EQ(gProfileStarted, true, "Currently the gPerf is disabled.");
  ProfilerStop();
  gProfileStarted = false;
#else
  LOG_S(WARNING) << "turbo_transformers is not compiled with gperftools.";
#endif
}

}  // namespace core
}  // namespace turbo_transformers
