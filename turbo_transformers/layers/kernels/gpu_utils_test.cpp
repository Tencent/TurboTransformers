

#include "loguru.hpp"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/utils.h"
#endif
#include "catch2/catch.hpp"
#include "turbo_transformers/core/enforce.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

#ifdef TT_WITH_CUDA
template <typename T, typename Func>
static void ConcatTestHelper(int batch_size, int dim1, int dim2,
                             int hidden_size, const Func& func) {
  core::Tensor cpu_t1(nullptr), cpu_t2(nullptr), cpu_out(nullptr),
      gpu_t1(nullptr), gpu_t2(nullptr), gpu_out(nullptr);
  std::tie(cpu_t1, gpu_t1) = common::CreateAndFillRandomForCPUGPUTensors<T>(
      {batch_size, 2, dim1, hidden_size});
  std::tie(cpu_t2, gpu_t2) = common::CreateAndFillRandomForCPUGPUTensors<T>(
      {batch_size, 2, dim2, hidden_size});
  func(cpu_t1, cpu_t2, cpu_out, gpu_t1, gpu_t2, gpu_out);
}

TEST_CASE("gpu-concat") {
  for (auto hidden_size : {16}) {
    for (auto batch_size : {1, 5}) {
      ConcatTestHelper<float>(
          batch_size, 7, 11, hidden_size,
          [](core::Tensor& cpu_t1, core::Tensor& cpu_t2, core::Tensor& cpu_out,
             core::Tensor& gpu_t1, core::Tensor& gpu_t2,
             core::Tensor& gpu_out) {
            kernels::Concat<float>(cpu_t1, cpu_t2, 2, &cpu_out);
            kernels::Concat<float>(gpu_t1, gpu_t2, 2, &gpu_out);
            REQUIRE(common::CheckResultOfCPUAndGPU<float>(cpu_out, gpu_out));
          });
    }
  }
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
