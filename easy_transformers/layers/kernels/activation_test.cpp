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
#include "easy_transformers/layers/kernels/activation.h"

#include <chrono>

#include "easy_transformers/core/half.h"
#ifdef FT_WITH_CUDA
#include "easy_transformers/core/cuda_device_context.h"
#endif
#include "catch2/catch.hpp"
#include "easy_transformers/core/aligned_scratchpad.h"
#include "easy_transformers/core/enforce.h"
#include "easy_transformers/layers/kernels/test_helper.h"
#include "loguru.hpp"

namespace easy_transformers {
namespace layers {
namespace kernels {

void AddBiasGeLUActNaive(const float* bias, float* out, int64_t m, int64_t n) {
  for (int64_t i = 0; i < m; ++i) {
    int64_t k = 0;
    for (int64_t j = n * i; j < n * (i + 1); ++j) {
      auto before_act = out[j] + bias[k++];
      out[j] = before_act * 0.5f *
               (1.0f + std::tanh(0.7978845608028654f *
                                 (before_act + 0.044715f * before_act *
                                                   before_act * before_act)));
    }
  }
}

class Timer {
 public:
  Timer() : start_(std::chrono::system_clock::now()) {}

  void Reset() { start_ = std::chrono::system_clock::now(); }

  double Elapse() {
    auto end = std::chrono::system_clock::now();
    ;
    auto duration = end - start_;
    return double(duration.count()) * std::chrono::microseconds::period::num /
           std::chrono::microseconds::period::den;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_;
};

template <typename Func>
void TestFunction(Func&& func, int step, const std::string& infor,
                  double g_bytes) {
  func();
  Timer timer;
  for (int i = 0; i < step; ++i) {
    func();
  }
  auto elapse = timer.Elapse() / step;

  LOG_S(INFO) << infor << " cost:" << elapse << " ms, Bandwidth "
              << g_bytes / elapse << " GB/s";
}

TEST_CASE("activation CPU benchmark") {
  auto tensor_create_and_fill_constant =
      [](std::initializer_list<int64_t> shape, float value) {
        easy_transformers::core::Tensor tensor(nullptr);
        tensor.Reshape<float>(shape, kDLCPU, 0);
        auto* ptr = tensor.mutableData<float>();
        for (int64_t i = 0; i < tensor.numel(); ++i) {
          ptr[i] = value;
        }
        return tensor;
      };

  int64_t hidden_size = 12 * 64;
  const int step = 10;
  for (auto batch_size : {1, 20, 24})
    for (auto seq_length : {8, 16, 32, 48, 64, 128}) {
      auto m = batch_size * seq_length;
      auto n = hidden_size;

      auto bias = tensor_create_and_fill_constant({n}, 0.01f);
      auto out = tensor_create_and_fill_constant({m, n}, 0.02f);
      auto out_parallel = tensor_create_and_fill_constant({m, n}, 0.02f);

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;

      TestFunction(
          [&]() {
            AddBiasGeLUActNaive(bias.data<float>(),
                                out_parallel.mutableData<float>(), m, n);
          },
          step, "AddBiasGeLUActNaive", m * n * sizeof(float) / 1e9);

      TestFunction([&]() { AddBiasGeLUAct<float>(bias, &out); }, step,
                   "AddBiasGeLUAct OMP", m * n * sizeof(float) / 1e9);

      auto* out_parallel_ptr = out_parallel.mutableData<float>();
      for (int64_t i = 0; i < m * n; ++i) {
        FT_ENFORCE_LT(fabs(out_parallel_ptr[i] - out_parallel_ptr[i]), 1e-6,
                      "Wrong @ %d", i);
      }
    }
}

#ifdef FT_WITH_CUDA

class GPUTimer {
 public:
  GPUTimer(cudaStream_t stream) : stream_(stream) {
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
    cudaEventRecord(start_event_, stream_);
  }

  double Elapse() {
    cudaEventRecord(stop_event_, stream_);
    cudaEventSynchronize(stop_event_);
    float elapse;
    cudaEventElapsedTime(&elapse, start_event_, stop_event_);
    elapse /= 1000;  // ms
    return elapse;
  }

 private:
  cudaEvent_t start_event_, stop_event_;
  cudaStream_t stream_;
};

template <typename T>
easy_transformers::core::Tensor CreateTensor(
    std::initializer_list<int64_t> shape, DLDeviceType device_type,
    int dev_id) {
  easy_transformers::core::Tensor tensor(nullptr);
  tensor.Reshape<T>(shape, device_type, dev_id);
  return tensor;
};

TEST_CASE("activation CPU and GPU correctness") {
  int64_t hidden_size = 12 * 64;

  std::vector<int64_t> batch_size_list{1, 20, 24};
  std::vector<int64_t> seq_length_list{8, 16, 32, 48, 64, 128};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      auto gpu_bias = CreateTensor<float>({hidden_size}, kDLGPU, 0);
      auto cpu_bias = CreateTensor<float>({hidden_size}, kDLCPU, 0);

      auto gpu_out =
          CreateTensor<float>({batch_size, seq_length, hidden_size}, kDLGPU, 0);
      auto cpu_out =
          CreateTensor<float>({batch_size, seq_length, hidden_size}, kDLCPU, 0);

      ::easy_transformers::test::FillDataForCPUGPUTensors<float>(cpu_bias,
                                                                 gpu_bias);
      ::easy_transformers::test::FillDataForCPUGPUTensors<float>(cpu_out,
                                                                 gpu_out);

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;
      AddBiasGeLUAct<float>(cpu_bias, &cpu_out);
      AddBiasGeLUAct<float>(gpu_bias, &gpu_out);
      REQUIRE(
          ::easy_transformers::test::CompareCPUGPU<float>(cpu_out, gpu_out));
    }  // for
}

template <typename T>
void ActivationGPUBenchmark(int batch_size, int seq_length, int hidden_size,
                            int step) {
  auto m = batch_size * seq_length;
  auto n = hidden_size;
  auto bias = CreateTensor<T>({hidden_size}, kDLGPU, 0);
  auto out = CreateTensor<T>({batch_size, seq_length, hidden_size}, kDLGPU, 0);
  ::easy_transformers::test::Fill<T>(out);
  ::easy_transformers::test::Fill<T>(bias);

  LOG_S(INFO) << "batch_size: " << batch_size << " seq_length: " << seq_length;
  AddBiasGeLUAct<T>(bias, &out);
  auto& cuda_ctx = easy_transformers::core::CUDADeviceContext::GetInstance();
  auto stream = cuda_ctx.stream();
  GPUTimer timer(stream);
  for (int i = 0; i < step; ++i) {
    AddBiasGeLUAct<T>(bias, &out);
  }
  auto elapse = timer.Elapse() / step * 1000;
  LOG_S(INFO) << "AddBiasGeLUAct GPU cost:" << elapse << " ms, Bandwidth "
              << m * n * 0.1 / 1e6 / elapse << " GB/s";
}

TEST_CASE("activation GPU benchmark") {
  int64_t hidden_size = 12 * 64;
  const int step = 10;
  for (auto batch_size : {1, 20, 24}) {
    for (auto seq_length : {8, 16, 32, 48, 64, 128}) {
      ActivationGPUBenchmark<float>(batch_size, seq_length, hidden_size, step);
      ActivationGPUBenchmark<core::Half>(batch_size, seq_length, hidden_size,
                                         step);
    }
  }
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace easy_transformers
