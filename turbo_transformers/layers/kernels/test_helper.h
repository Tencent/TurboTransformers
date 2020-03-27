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

#include "turbo_transformers/core/memory.h"

namespace turbo_transformers {
namespace test {

using Tensor = turbo_transformers::core::Tensor;

#ifdef FT_WITH_CUDA
template <typename T>
void FillDataForCPUGPUTensors(Tensor& cpu_tensor, Tensor& gpu_tensor) {
  T* gpu_data = gpu_tensor.mutableData<T>();
  T* cpu_data = cpu_tensor.mutableData<T>();
  auto size = cpu_tensor.numel();
  srand((unsigned)time(NULL));
  for (int64_t i = 0; i < size; ++i) {
    cpu_data[i] = rand() / static_cast<T>(RAND_MAX);
  }
  turbo_transformers::core::Memcpy(
      gpu_data, cpu_data, size * sizeof(T),
      ::turbo_transformers::core::MemcpyFlag::kCPU2GPU);
}

template <typename T>
bool CheckResultOfCPUAndGPU(const Tensor& cpu_tensor,
                            const Tensor& gpu_tensor) {
  const T* gpu_data = gpu_tensor.data<T>();
  const T* cpu_data = cpu_tensor.data<T>();
  auto size = cpu_tensor.numel();

  std::unique_ptr<T[]> gpu_data_ref(new T[size]);
  turbo_transformers::core::Memcpy(
      gpu_data_ref.get(), gpu_data, size * sizeof(T),
      turbo_transformers::core::MemcpyFlag::kGPU2CPU);
  bool ret = true;
  for (int64_t i = 0; i < size; ++i) {
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

template <typename T>
inline void RandomFillHost(T* m, const int mSize, float LO = 0.,
                           float HI = 1.) {
  srand(static_cast<unsigned>(time(0)));
  for (int i = 0; i < mSize; i++)
    m[i] = LO +
           static_cast<T>(rand()) / (static_cast<float>(RAND_MAX / (HI - LO)));
}

template <typename T>
void Fill(turbo_transformers::core::Tensor& tensor, T lower_bound = 0.,
          T upper_bound = 1.) {
  T* T_data = tensor.mutableData<T>();
  auto size = tensor.numel();
  std::unique_ptr<T> cpu_data(new T[size]);
  RandomFillHost(cpu_data.get(), size, lower_bound, upper_bound);

  turbo_transformers::core::MemcpyFlag flag;
  if (tensor.device_type() == kDLCPU) {
    flag = turbo_transformers::core::MemcpyFlag::kCPU2CPU;
  } else if (tensor.device_type() == kDLGPU) {
    flag = turbo_transformers::core::MemcpyFlag::kCPU2GPU;
  } else {
    FT_THROW("Fill device_type wrong");
  }
  ::turbo_transformers::core::Memcpy(T_data, cpu_data.get(), size * sizeof(T),
                                     flag);
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
