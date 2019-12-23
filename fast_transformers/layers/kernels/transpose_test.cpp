#define CATCH_CONFIG_MAIN
#include "fast_transformers/layers/kernels/transpose.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/blas.h"
#include "fast_transformers/core/enforce.h"
#include "fast_transformers/layers/kernels/test_helper.h"
#include "loguru.hpp"

namespace fast_transformers {
namespace layers {
namespace kernels {

#ifdef FT_WITH_CUDA
TEST_CASE("split_add_transpose CPU and GPU correctness") {
  int64_t num_attention_heads = 12;

  std::vector<int64_t> batch_size_list{1, 20, 24};
  std::vector<int64_t> seq_length_list{10, 20, 32, 64, 128};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      fast_transformers::core::Tensor input_tensor_gpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, 3, num_attention_heads, 64}, kDLGPU, 0));
      fast_transformers::core::Tensor input_tensor_cpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, 3, num_attention_heads, 64}, kDLCPU, 0));

      ::fast_transformers::test::FillCPUGPU<float>(input_tensor_cpu,
                                                   input_tensor_gpu);

      fast_transformers::core::Tensor bias_tensor_gpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {3, num_attention_heads, 64}, kDLGPU, 0));

      fast_transformers::core::Tensor bias_tensor_cpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {3, num_attention_heads, 64}, kDLCPU, 0));

      ::fast_transformers::test::FillCPUGPU<float>(bias_tensor_cpu,
                                                   bias_tensor_gpu);

      fast_transformers::core::Tensor output_tensor_gpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {3, batch_size, num_attention_heads, seq_length, 64}, kDLGPU, 0));
      fast_transformers::core::Tensor output_tensor_cpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {3, batch_size, num_attention_heads, seq_length, 64}, kDLCPU, 0));

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;

      SplitAddBiasTransposeForScore(&output_tensor_gpu, input_tensor_gpu,
                                    bias_tensor_gpu);
      SplitAddBiasTransposeForScore(&output_tensor_cpu, input_tensor_cpu,
                                    bias_tensor_cpu);
      REQUIRE(::fast_transformers::test::CompareCPUGPU<float>(
          output_tensor_cpu, output_tensor_gpu));
    }
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
