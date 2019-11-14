#include "profiler.h"
#include <glog/logging.h>
#ifdef WITH_GPERFTOOLS
#include "gperftools/profiler.h"
#endif

namespace fast_transformers {
namespace core {
static std::once_flag gProfileOnce;
#ifdef WITH_GPERFTOOLS
static bool gProfileStarted = false;
#endif

void EnableGPerf(const std::string& profile_file) {
#ifdef WITH_GPERFTOOLS
  VLOG(1) << "gperf tools enabled." << profile_file;
    std::call_once(gProfileOnce, [&] {
#ifdef WITH_GPERFTOOLS
      ProfilerStart(profile_file.c_str());
      gProfileStarted = true;
#else
      LOG(WARNING) << "fast_transformers is not compiled with gperftools.";
#endif
    });
#endif
}

void DisableGPerf(){

}

}  // namespace core
}  // namespace fast_transformers
