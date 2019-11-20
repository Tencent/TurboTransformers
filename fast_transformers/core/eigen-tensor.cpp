#include "fast_transformers/core/eigen-tensor.h"
#include <thread>
#include <unsupported/Eigen/CXX11/ThreadPool>

namespace fast_transformers {
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
}  // namespace fast_transformers
