#define CATCH_CONFIG_MAIN
#include "fast_transformers/layers/kernels/layer_norm.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/enforce.h"
#include "fast_transformers/layers/kernels/test_helper.h"
#include "loguru.hpp"

namespace fast_transformers {
namespace layers {
namespace kernels {

#ifdef FT_WITH_CUDA
TEST_CASE("layer_norm CPU and GPU correctness") {
  int64_t hidden_size = 12 * 64;

  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{8, 16, 32, 48, 64, 128};
  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      fast_transformers::core::Tensor gpu_gamma(
          fast_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                           kDLGPU, 0));

      fast_transformers::core::Tensor cpu_gamma(
          fast_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                           kDLCPU, 0));

      fast_transformers::core::Tensor gpu_beta(
          fast_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                           kDLGPU, 0));

      fast_transformers::core::Tensor cpu_beta(
          fast_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                           kDLCPU, 0));

      fast_transformers::core::Tensor gpu_out(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLGPU, 0));

      fast_transformers::core::Tensor cpu_out(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLCPU, 0));

      ::fast_transformers::test::FillDataForCPUGPUTensors<float>(cpu_gamma,
                                                                 gpu_gamma);
      ::fast_transformers::test::FillDataForCPUGPUTensors<float>(cpu_beta,
                                                                 gpu_beta);
      ::fast_transformers::test::FillDataForCPUGPUTensors<float>(cpu_out,
                                                                 gpu_out);

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;
      {
        LayerNorm<float>(cpu_gamma, cpu_beta, &cpu_out);
        LayerNorm<float>(gpu_gamma, gpu_beta, &gpu_out);
      }
      REQUIRE(
          ::fast_transformers::test::CompareCPUGPU<float>(cpu_out, gpu_out));
    }  // for
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
