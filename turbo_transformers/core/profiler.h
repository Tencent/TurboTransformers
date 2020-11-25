

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
