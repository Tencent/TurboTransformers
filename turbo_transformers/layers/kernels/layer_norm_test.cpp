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

#include "turbo_transformers/layers/kernels/layer_norm.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "loguru.hpp"
#include "turbo_transformers/layers/kernels/test_helper.h"

#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {

#ifdef TT_WITH_CUDA
TEST_CASE("add_bias_layer_norm CPU and GPU correctness") {
  int64_t hidden_size = 12 * 64;

  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{10,  20,  40,  60,  80,
                                       100, 200, 300, 400, 500};
  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      core::Tensor gpu_input(nullptr), cpu_input(nullptr), gpu_bias(nullptr),
          cpu_bias(nullptr), gpu_out(nullptr), cpu_out(nullptr),
          gpu_gamma(nullptr), cpu_gamma(nullptr), gpu_beta(nullptr),
          cpu_beta(nullptr);
      std::tie(cpu_input, gpu_input) =
          test::CreateAndFillRandomForCPUGPUTensors<float>(
              {batch_size, seq_length, hidden_size});
      std::tie(cpu_bias, gpu_bias) =
          test::CreateAndFillRandomForCPUGPUTensors<float>({hidden_size});
      std::tie(cpu_out, gpu_out) =
          test::CreateAndFillRandomForCPUGPUTensors<float>(
              {batch_size, seq_length, hidden_size});
      std::tie(cpu_gamma, gpu_gamma) =
          test::CreateAndFillRandomForCPUGPUTensors<float>({hidden_size});
      std::tie(cpu_beta, gpu_beta) =
          test::CreateAndFillRandomForCPUGPUTensors<float>({hidden_size});

      std::cout << "batch_size: " << batch_size
                << " seq_length: " << seq_length;
      {
        LayerNorm<float>(cpu_gamma, cpu_beta, &cpu_out);
        LayerNorm<float>(gpu_gamma, gpu_beta, &gpu_out);
      }
      REQUIRE(test::CheckResultOfCPUAndGPU<float>(cpu_out, gpu_out));
      {
        AddBiasLayerNorm<float>(cpu_input, cpu_bias, cpu_gamma, cpu_beta,
                                &cpu_out);
        AddBiasLayerNorm<float>(gpu_input, gpu_bias, gpu_gamma, gpu_beta,
                                &gpu_out);
      }
      REQUIRE(test::CheckResultOfCPUAndGPU<float>(cpu_out, gpu_out));

      // WARM UP
      for (int i = 0; i < 5; ++i) {
        AddBiasLayerNorm<float>(gpu_input, gpu_bias, gpu_gamma, gpu_beta,
                                &gpu_out);
      }

      cudaEvent_t start_event, stop_event;
      cudaEventCreate(&start_event);
      cudaEventCreate(&stop_event);
      auto& cuda_ctx = core::CUDADeviceContext::GetInstance();
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
      auto cpu_input = test::CreateTensorAndFillRandom<float>(
          {batch_size, seq_length, hidden_size}, kDLCPU, 0);
      auto cpu_bias =
          test::CreateTensorAndFillRandom<float>({hidden_size}, kDLCPU, 0);
      auto cpu_gamma =
          test::CreateTensorAndFillRandom<float>({hidden_size}, kDLCPU, 0);
      auto cpu_beta =
          test::CreateTensorAndFillRandom<float>({hidden_size}, kDLCPU, 0);
      auto cpu_out = test::CreateTensorAndFillRandom<float>(
          {batch_size, seq_length, hidden_size}, kDLCPU, 0);

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
