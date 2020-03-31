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
#include <chrono>
#include <tuple>

#include "loguru.hpp"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/core/tensor_copy.h"
#include "turbo_transformers/layers/kernels/common.h"

namespace turbo_transformers {
namespace test {

using Tensor = core::Tensor;

template <typename T>
void RandomFillHost(T* m, size_t size) {
  srand(static_cast<unsigned>(time(nullptr)));
  for (size_t i = 0; i < size; i++) {
    m[i] = static_cast<T>(rand() / static_cast<T>(RAND_MAX));
  }
}

template <typename T>
Tensor CreateTensor(std::initializer_list<int64_t> shape,
                    DLDeviceType device_type, int dev_id) {
  Tensor tensor(nullptr);
  tensor.Reshape<T>(shape, device_type, dev_id);

  return tensor;
}

#ifdef TT_WITH_CUDA
template <typename T>
std::tuple<Tensor, Tensor> CreateAndFillRandomForCPUGPUTensors(
    std::initializer_list<int64_t> shape) {
  Tensor cpu_tensor = CreateTensor<T>(shape, kDLCPU, 0);
  Tensor gpu_tensor = CreateTensor<T>(shape, kDLGPU, 0);
  RandomFillHost(cpu_tensor.mutableData<T>(), cpu_tensor.numel());
  core::Copy<T>(cpu_tensor, gpu_tensor);
  return std::make_tuple(std::move(cpu_tensor), std::move(gpu_tensor));
}

template <typename T>
bool CheckResultOfCPUAndGPU(const Tensor& cpu_tensor,
                            const Tensor& gpu_tensor) {
  TT_ENFORCE(layers::kernels::common::is_same_shape(cpu_tensor, gpu_tensor),
             "The shape of the inputs is not equal.");
  const T* cpu_data = cpu_tensor.data<T>();
  Tensor tmp_tensor = CreateTensor<T>({gpu_tensor.numel()}, kDLCPU, 0);
  core::Copy<T>(gpu_tensor, tmp_tensor);
  const T* gpu_data_ref = tmp_tensor.data<T>();
  bool ret = true;
  for (int64_t i = 0; i < gpu_tensor.numel(); ++i) {
    if (std::abs(gpu_data_ref[i] - cpu_data[i]) > 1e-3) {
      std::cerr << "@ " << i << ": " << gpu_data_ref[i] << " vs " << cpu_data[i]
                << std::endl;
      ret = false;
      break;
    }
  }
  return ret;
}

class GPUTimer {
 public:
  explicit GPUTimer(cudaStream_t stream) : stream_(stream) {
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
    cudaEventRecord(start_event_, stream_);
  }

  double ElapseSecond() {
    cudaEventRecord(stop_event_, stream_);
    cudaEventSynchronize(stop_event_);
    float elapse;
    cudaEventElapsedTime(&elapse, start_event_, stop_event_);
    return elapse / 1000;
  }

 private:
  cudaEvent_t start_event_, stop_event_;
  cudaStream_t stream_;
};
#endif

class Timer {
 public:
  Timer() : start_(std::chrono::system_clock::now()) {}

  void Reset() { start_ = std::chrono::system_clock::now(); }

  double ElapseSecond() {
    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
    return double(duration.count()) * std::chrono::microseconds::period::num /
           std::chrono::microseconds::period::den;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_;
};

template <typename Func>
void TestFuncSpeed(Func&& func, int step, const std::string& infor,
                   double g_bytes) {
  func();
  test::Timer timer;
  for (int i = 0; i < step; ++i) {
    func();
  }
  auto elapse = timer.ElapseSecond() / step;

  LOG_S(INFO) << infor << " cost:" << elapse << " ms, Bandwidth "
              << g_bytes / elapse << " GB/s";
}

template <typename T>
Tensor CreateTensorAndFillConstant(std::initializer_list<int64_t> shape,
                                   DLDeviceType dev_type, int dev_id, T value) {
  Tensor tensor = CreateTensor<T>(shape, dev_type, dev_id);
  layers::kernels::common::Fill<T>(tensor.mutableData<T>(), tensor.numel(),
                                   value, dev_type);
  return tensor;
}

template <typename T>
Tensor CreateTensorAndFillRandom(std::initializer_list<int64_t> shape,
                                 DLDeviceType dev_type, int dev_id) {
  Tensor cpu_tensor = CreateTensor<T>(shape, kDLCPU, dev_id);
  RandomFillHost(cpu_tensor.mutableData<T>(), cpu_tensor.numel());
  if (dev_type == kDLGPU) {
    Tensor gpu_tensor = CreateTensor<T>(shape, dev_type, dev_id);
    core::Copy<T>(cpu_tensor, gpu_tensor);
    return gpu_tensor;
  }
  return cpu_tensor;
}

template <typename T>
void Fill(Tensor& tensor) {
  T* T_data = tensor.mutableData<T>();
  auto size = tensor.numel();
  std::unique_ptr<T> cpu_data(new T[size]);
  RandomFillHost(cpu_data.get(), size);

  core::MemcpyFlag flag;
  if (tensor.device_type() == kDLCPU) {
    flag = core::MemcpyFlag::kCPU2CPU;
  } else if (tensor.device_type() == kDLGPU) {
    flag = core::MemcpyFlag::kCPU2GPU;
  } else {
    TT_THROW("Fill device_type wrong");
  }
  core::Memcpy(T_data, cpu_data.get(), size * sizeof(T), flag);
}

template <typename T>
bool CheckResultOfCPU(const Tensor& cpu_tensor_lhs,
                      const Tensor& cpu_tensor_rhs) {
  const T* cpu_data_lhs = cpu_tensor_lhs.data<T>();
  const T* cpu_data_rhs = cpu_tensor_rhs.data<T>();
  auto size = cpu_tensor_lhs.numel();

  bool ret = true;
  for (int64_t i = 0; i < size; ++i) {
    if (std::abs(cpu_data_lhs[i] - cpu_data_rhs[i]) > 1e-3) {
      std::cerr << "@ " << i << ": " << cpu_data_lhs[i] << " vs "
                << cpu_data_rhs[i] << std::endl;
      ret = false;
      break;
    }
  }
  return ret;
}

}  // namespace test
}  // namespace turbo_transformers
