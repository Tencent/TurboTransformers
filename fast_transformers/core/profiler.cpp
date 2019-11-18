#include "profiler.h"
#include "enforce.h"
#include "loguru.hpp"

#ifdef WITH_GPERFTOOLS
#include "gperftools/profiler.h"
#endif

namespace fast_transformers {
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
  LOG_S(WARNING) << "fast_transformers is not compiled with gperftools.";
#endif
}

void DisableGperf() {
#ifdef WITH_GPERFTOOLS
  FT_ENFORCE_EQ(gProfileStarted, true, "Currently the gPerf is disabled.");
  ProfilerStop();
  gProfileStarted = false;
#else
  LOG_S(WARNING) << "fast_transformers is not compiled with gperftools.";
#endif
}

}  // namespace core
}  // namespace fast_transformers
