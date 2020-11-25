

#include "turbo_transformers/layers/kernels/transpose.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "loguru.hpp"
#include "turbo_transformers/core/blas.h"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/layers/kernels/common.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

#ifdef TT_WITH_CUDA
TEST_CASE("splitaddtranspose-gpu-test") {
  const std::vector<int64_t> num_attention_heads_list{12};
  const std::vector<int64_t> batch_size_list{1, 12, 20};
  const std::vector<int64_t> seq_length_list{10, 20, 32, 64, 128};

  for (auto hidden_size : {64, 2000})
    for (auto num_attention_heads : num_attention_heads_list)
      for (auto batch_size : batch_size_list)
        for (auto seq_length : seq_length_list) {
          core::Tensor input_tensor_cpu(nullptr), input_tensor_gpu(nullptr);
          std::tie(input_tensor_cpu, input_tensor_gpu) =
              common::CreateAndFillRandomForCPUGPUTensors<float>(
                  {batch_size, seq_length, 3, num_attention_heads,
                   hidden_size});

          core::Tensor bias_tensor_cpu(nullptr), bias_tensor_gpu(nullptr);
          std::tie(bias_tensor_cpu, bias_tensor_gpu) =
              common::CreateAndFillRandomForCPUGPUTensors<float>(
                  {3, num_attention_heads, hidden_size});

          turbo_transformers::core::Tensor output_tensor_gpu(
              turbo_transformers::core::NewDLPackTensorT<float>(
                  {3, batch_size, num_attention_heads, seq_length, hidden_size},
                  kDLGPU, 0));
          turbo_transformers::core::Tensor output_tensor_cpu(
              turbo_transformers::core::NewDLPackTensorT<float>(
                  {3, batch_size, num_attention_heads, seq_length, hidden_size},
                  kDLCPU, 0));

          SplitAddBiasTransposeForScore(&output_tensor_gpu, input_tensor_gpu,
                                        bias_tensor_gpu);
          SplitAddBiasTransposeForScore(&output_tensor_cpu, input_tensor_cpu,
                                        bias_tensor_cpu);
          REQUIRE(common::CheckResultOfCPUAndGPU<float>(output_tensor_cpu,
                                                        output_tensor_gpu));
        }
}

TEST_CASE("transpose-gpu-test") {
  const std::vector<int64_t> num_attention_heads_list{12, 20, 24};
  const std::vector<int64_t> batch_size_list{
      1,
      20,
  };
  const std::vector<int64_t> seq_length_list{10, 32, 128};

  for (auto num_attention_heads : num_attention_heads_list)
    for (auto batch_size : batch_size_list)
      for (auto seq_length : seq_length_list) {
        core::Tensor input_tensor_cpu(nullptr), input_tensor_gpu(nullptr);
        std::tie(input_tensor_cpu, input_tensor_gpu) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {batch_size, num_attention_heads, seq_length, 64});

        turbo_transformers::core::Tensor output_tensor_gpu(
            turbo_transformers::core::NewDLPackTensorT<float>(
                {batch_size, seq_length, num_attention_heads, 64}, kDLGPU, 0));
        turbo_transformers::core::Tensor output_tensor_cpu(
            turbo_transformers::core::NewDLPackTensorT<float>(
                {batch_size, seq_length, num_attention_heads, 64}, kDLCPU, 0));

        TransposeForScore(&output_tensor_gpu, input_tensor_gpu);
        TransposeForScore(&output_tensor_cpu, input_tensor_cpu);
        REQUIRE(common::CheckResultOfCPUAndGPU<float>(output_tensor_cpu,
                                                      output_tensor_gpu));
      }
}

TEST_CASE("transpose-bias-gpu-test") {
  const std::vector<int64_t> num_attention_heads_list{12, 20, 24};
  const std::vector<int64_t> batch_size_list{
      1,
      20,
  };
  const std::vector<int64_t> seq_length_list{10, 32, 128};

  for (auto num_attention_heads : num_attention_heads_list)
    for (auto batch_size : batch_size_list)
      for (auto seq_length : seq_length_list) {
        core::Tensor input_tensor_cpu(nullptr), input_tensor_gpu(nullptr);
        core::Tensor bias_tensor_cpu(nullptr), bias_tensor_gpu(nullptr);
        std::tie(input_tensor_cpu, input_tensor_gpu) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {batch_size, seq_length, num_attention_heads, 64});
        std::tie(bias_tensor_cpu, bias_tensor_gpu) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {num_attention_heads, 64});

        turbo_transformers::core::Tensor output_tensor_gpu(
            turbo_transformers::core::NewDLPackTensorT<float>(
                {batch_size, num_attention_heads, seq_length, 64}, kDLGPU, 0));
        turbo_transformers::core::Tensor output_tensor_cpu(
            turbo_transformers::core::NewDLPackTensorT<float>(
                {batch_size, num_attention_heads, seq_length, 64}, kDLCPU, 0));

        AddBiasTransposeForScore(input_tensor_gpu, bias_tensor_gpu,
                                 &output_tensor_gpu);
        AddBiasTransposeForScore(input_tensor_cpu, bias_tensor_cpu,
                                 &output_tensor_cpu);
        REQUIRE(common::CheckResultOfCPUAndGPU<float>(output_tensor_cpu,
                                                      output_tensor_gpu));
      }
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
