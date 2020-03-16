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

#define CATCH_CONFIG_MAIN

#include "turbo_transformers/layers/kernels/layer_norm.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "loguru.hpp"
#include "turbo_transformers/core/aligned_scratchpad.h"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/layers/kernels/test_helper.h"

#ifdef FT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {

#ifdef FT_WITH_CUDA
TEST_CASE("layer_norm CPU and GPU correctness") {
  int64_t hidden_size = 12 * 64;

  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{10, 20, 32, 48, 64, 128};
  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      turbo_transformers::core::Tensor gpu_gamma(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLGPU, 0));

      turbo_transformers::core::Tensor cpu_gamma(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLCPU, 0));

      turbo_transformers::core::Tensor gpu_beta(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLGPU, 0));

      turbo_transformers::core::Tensor cpu_beta(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLCPU, 0));

      turbo_transformers::core::Tensor gpu_out(
          turbo_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLGPU, 0));

      turbo_transformers::core::Tensor cpu_out(
          turbo_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLCPU, 0));

      ::turbo_transformers::test::FillDataForCPUGPUTensors<float>(cpu_gamma,
                                                                  gpu_gamma);
      ::turbo_transformers::test::FillDataForCPUGPUTensors<float>(cpu_beta,
                                                                  gpu_beta);
      ::turbo_transformers::test::FillDataForCPUGPUTensors<float>(cpu_out,
                                                                  gpu_out);

      std::cout << batch_size << ", seq_length" << seq_length << std::endl;
      {
        LayerNorm<float>(cpu_gamma, cpu_beta, &cpu_out);
        LayerNorm<float>(gpu_gamma, gpu_beta, &gpu_out);
      }
      REQUIRE(
          ::turbo_transformers::test::CompareCPUGPU<float>(cpu_out, gpu_out));
    }  // for
}

TEST_CASE("add_bias_layer_norm CPU and GPU correctness") {
  int64_t hidden_size = 12 * 64;

  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{10,  20,  40,  60,  80,
                                       100, 200, 300, 400, 500};
  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      turbo_transformers::core::Tensor gpu_input(
          turbo_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLGPU, 0));
      turbo_transformers::core::Tensor cpu_input(
          turbo_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLCPU, 0));

      turbo_transformers::core::Tensor gpu_bias(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLGPU, 0));
      turbo_transformers::core::Tensor cpu_bias(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLCPU, 0));

      turbo_transformers::core::Tensor gpu_gamma(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLGPU, 0));
      turbo_transformers::core::Tensor cpu_gamma(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLCPU, 0));

      turbo_transformers::core::Tensor gpu_beta(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLGPU, 0));
      turbo_transformers::core::Tensor cpu_beta(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLCPU, 0));

      turbo_transformers::core::Tensor gpu_out(
          turbo_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLGPU, 0));
      turbo_transformers::core::Tensor cpu_out(
          turbo_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLCPU, 0));

      ::turbo_transformers::test::FillDataForCPUGPUTensors<float>(cpu_input,
                                                                  gpu_input);
      ::turbo_transformers::test::FillDataForCPUGPUTensors<float>(cpu_bias,
                                                                  gpu_bias);
      ::turbo_transformers::test::FillDataForCPUGPUTensors<float>(cpu_gamma,
                                                                  gpu_gamma);
      ::turbo_transformers::test::FillDataForCPUGPUTensors<float>(cpu_beta,
                                                                  gpu_beta);
      ::turbo_transformers::test::FillDataForCPUGPUTensors<float>(cpu_out,
                                                                  gpu_out);

      std::cout << "batch_size: " << batch_size
                << " seq_length: " << seq_length;
      {
        AddBiasLayerNorm<float>(cpu_input, cpu_bias, cpu_gamma, cpu_beta,
                                &cpu_out);
        AddBiasLayerNorm<float>(gpu_input, gpu_bias, gpu_gamma, gpu_beta,
                                &gpu_out);
      }
      REQUIRE(
          ::turbo_transformers::test::CompareCPUGPU<float>(cpu_out, gpu_out));

      // WARM UP
      for (int i = 0; i < 5; ++i) {
        AddBiasLayerNorm<float>(gpu_input, gpu_bias, gpu_gamma, gpu_beta,
                                &gpu_out);
      }

      cudaEvent_t start_event, stop_event;
      cudaEventCreate(&start_event);
      cudaEventCreate(&stop_event);
      auto& cuda_ctx =
          turbo_transformers::core::CUDADeviceContext::GetInstance();
      auto stream = cuda_ctx.stream();
      cudaEventRecord(start_event, stream);

      int step = 150;
      for (int i = 0; i < step; ++i) {
        AddBiasLayerNorm<float>(gpu_input, gpu_bias, gpu_gamma, gpu_beta,
                                &gpu_out);
      }

      cudaEventRecord(stop_event, stream);
      cudaEventSynchronize(stop_event);
      float elapse;
      cudaEventElapsedTime(&elapse, start_event, stop_event);
      elapse /= step;
      elapse /= 1000;  // ms

      std::cout << "AddBiasLayerNorm gpu cost, "
                << batch_size * seq_length * hidden_size * sizeof(float) / 1e9 /
                       elapse
                << ", GB/s, time consumed, " << elapse << std::endl;
    }  // for
}
#endif

TEST_CASE("add_bias_layer_norm CPU benchmark") {
  int64_t hidden_size = 12 * 64;

  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{10,  20,  40,  60,  80,
                                       100, 200, 300, 400, 500};
  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      turbo_transformers::core::Tensor cpu_input(
          turbo_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLCPU, 0));

      turbo_transformers::core::Tensor cpu_bias(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLCPU, 0));

      turbo_transformers::core::Tensor cpu_gamma(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLCPU, 0));

      turbo_transformers::core::Tensor cpu_beta(
          turbo_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                            kDLCPU, 0));

      turbo_transformers::core::Tensor cpu_out(
          turbo_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLCPU, 0));

      test::RandomFillHost(cpu_input.mutableData<float>(), cpu_input.numel());
      test::RandomFillHost(cpu_bias.mutableData<float>(), cpu_bias.numel());
      test::RandomFillHost(cpu_gamma.mutableData<float>(), cpu_gamma.numel());
      test::RandomFillHost(cpu_beta.mutableData<float>(), cpu_beta.numel());
      test::RandomFillHost(cpu_out.mutableData<float>(), cpu_out.numel());

      std::cout << "batch_size: " << batch_size << " seq_length: " << seq_length
                << " ";

      int step = 150;
      auto start = std::chrono::system_clock::now();
      for (int i = 0; i < step; ++i) {
        AddBiasLayerNorm<float>(cpu_input, cpu_bias, cpu_gamma, cpu_beta,
                                &cpu_out);
      }
      auto end = std::chrono::system_clock::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      auto elapse = double(duration.count()) *
                    std::chrono::microseconds::period::num /
                    std::chrono::microseconds::period::den / step;
      std::cout << "CPU AddBiasLayerNorm cost,"
                << batch_size * seq_length * hidden_size * sizeof(float) / 1e9 /
                       elapse
                << ", GB/s" << std::endl;
    }  // for
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
