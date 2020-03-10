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

#include "fast_transformers/layers/kernels/softmax.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/blas.h"
#include "fast_transformers/core/enforce.h"
#include "fast_transformers/layers/kernels/test_helper.h"
#include "loguru.hpp"
#ifdef FT_WITH_CUDA
#include "fast_transformers/core/cuda_device_context.h"
#endif

namespace fast_transformers {
namespace layers {
namespace kernels {

inline void _CreateBenchmark(DLDeviceType device_type) {
  const std::string device_name = device_type == kDLCPU ? "CPU" : "GPU";
  const int step = 100;

  int64_t num_attention_heads = 12;
  constexpr float scaler = 1.;

  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{10,  20,  40,  60,  80,
                                       100, 200, 300, 400, 500};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      fast_transformers::core::Tensor qk_buf_tensor(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, num_attention_heads, seq_length, seq_length},
              device_type, 0));
      ::fast_transformers::test::Fill<float>(qk_buf_tensor);
      fast_transformers::core::Tensor attr_mask_tensor(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length}, device_type, 0));
      ::fast_transformers::test::Fill<float>(attr_mask_tensor);

      auto data_size =
          batch_size * num_attention_heads * seq_length * seq_length;

      std::cout << "batch_size: " << batch_size
                << " seq_length: " << seq_length;

      ApplyMaskAndSoftmax(&qk_buf_tensor, attr_mask_tensor, scaler);

      // WARM UP
      for (int i = 0; i < 2; ++i) {
        ApplyMaskAndSoftmax(&qk_buf_tensor, attr_mask_tensor, scaler);
      }

      auto start = std::chrono::system_clock::now();
      if (device_type == kDLGPU) {
#ifdef FT_WITH_CUDA
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        auto& cuda_ctx =
            fast_transformers::core::CUDADeviceContext::GetInstance();
        auto stream = cuda_ctx.stream();
        cudaEventRecord(start_event, stream);

        for (int i = 0; i < step; ++i) {
          ApplyMaskAndSoftmax(&qk_buf_tensor, attr_mask_tensor, scaler);
        }

        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        {
          float elapse;
          cudaEventElapsedTime(&elapse, start_event, stop_event);
          elapse /= step;
          elapse /= 1000;  // sec
          std::cout << ", " << device_name << ", SoftmaxMask throughput, "
                    << data_size * sizeof(float) / 1e9 / elapse << ", GB/s "
                    << std::endl;
        }
        continue;
#else
        throw std::runtime_error("Not compile with GPU.");
#endif
      } else {
        for (int i = 0; i < step; ++i) {
          ApplyMaskAndSoftmax(&qk_buf_tensor, attr_mask_tensor, scaler);
        }
      }

      auto end = std::chrono::system_clock::system_clock::now();
      auto duration_parallel =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      auto elapse = double(duration_parallel.count()) *
                    std::chrono::microseconds::period::num /
                    std::chrono::microseconds::period::den / step * 1000;

      std::cout << ", " << device_name << ", SoftmaxMask throughput, "
                << data_size * sizeof(float) / 1e6 / elapse << ", GB/s"
                << std::endl;
    }
}

#ifdef FT_WITH_CUDA
TEST_CASE("softmax CPU and GPU correctness") {
  int64_t num_attention_heads = 12;

  constexpr float scaler = 1.;

  static core::AlignedScratchpad<float> buf;
  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{10,  20,  40,  60,  80,
                                       100, 200, 300, 400, 500};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      fast_transformers::core::Tensor qk_buf_gpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, num_attention_heads, seq_length, seq_length}, kDLGPU,
              0));

      fast_transformers::core::Tensor qk_buf_cpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, num_attention_heads, seq_length, seq_length}, kDLCPU,
              0));
      ::fast_transformers::test::FillDataForCPUGPUTensors<float>(qk_buf_cpu,
                                                                 qk_buf_gpu);

      fast_transformers::core::Tensor attr_mask_gpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length}, kDLGPU, 0));

      fast_transformers::core::Tensor attr_mask_cpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length}, kDLCPU, 0));

      ::fast_transformers::test::FillDataForCPUGPUTensors<float>(attr_mask_cpu,
                                                                 attr_mask_gpu);

      ApplyMaskAndSoftmax(&qk_buf_gpu, attr_mask_gpu, scaler);

      ApplyMaskAndSoftmax(&qk_buf_cpu, attr_mask_cpu, scaler);

      REQUIRE(::fast_transformers::test::CompareCPUGPU<float>(qk_buf_cpu,
                                                              qk_buf_gpu));
    }
}

TEST_CASE("softmax GPU benchmark") { _CreateBenchmark(kDLGPU); }
#endif

// TEST_CASE("softmax CPU benchmark") { _CreateBenchmark(kDLCPU); }

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
