// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

#pragma once
#include <dlpack/dlpack.h>

#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/core/tensor_copy.h"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/layers/kernels/gpu_utils.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {
namespace common {

template <typename T>
void RandomFillHost(T* m, size_t size) {
  srand(static_cast<unsigned>(time(nullptr)));
  for (size_t i = 0; i < size; i++) {
    m[i] = static_cast<T>(rand() / static_cast<T>(RAND_MAX));
  }
}

extern bool is_same_device_ctx(DLContext t1, DLContext t2);

extern bool is_same_shape(const core::Tensor& t1, const core::Tensor& t2);

template <typename T>
void Sequence(T* data, int64_t size, DLDeviceType device);

template <typename T>
void Fill(T* data, int64_t size, T val, DLDeviceType device);

// TODO(jiaruifang): this function should better pass a function in.
// how can we pass a lambda function as __device__ to cuda?
void Transform(int64_t* src_data, float* dst_data, int64_t size,
               DLDeviceType device);

template <typename T>
void FillRandom(core::Tensor& tensor) {
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
core::Tensor CreateTensor(std::initializer_list<int64_t> shape,
                          DLDeviceType device_type, int dev_id) {
  core::Tensor tensor(nullptr);
  tensor.Reshape<T>(shape, device_type, dev_id);

  return tensor;
}

template <typename T>
core::Tensor CreateTensorAndFillConstant(std::initializer_list<int64_t> shape,
                                         DLDeviceType dev_type, int dev_id,
                                         T value) {
  core::Tensor tensor = CreateTensor<T>(shape, dev_type, dev_id);
  layers::kernels::common::Fill<T>(tensor.mutableData<T>(), tensor.numel(),
                                   value, dev_type);
  return tensor;
}

template <typename T>
core::Tensor CreateTensorAndFillRandom(std::initializer_list<int64_t> shape,
                                       DLDeviceType dev_type, int dev_id) {
  core::Tensor cpu_tensor = CreateTensor<T>(shape, kDLCPU, dev_id);
  RandomFillHost(cpu_tensor.mutableData<T>(), cpu_tensor.numel());
  if (dev_type == kDLGPU) {
    core::Tensor gpu_tensor = CreateTensor<T>(shape, dev_type, dev_id);
    core::Copy<T>(cpu_tensor, gpu_tensor);
    return gpu_tensor;
  }
  return cpu_tensor;
}

#ifdef TT_WITH_CUDA
template <typename T>
std::tuple<core::Tensor, core::Tensor> CreateAndFillRandomForCPUGPUTensors(
    std::initializer_list<int64_t> shape) {
  core::Tensor cpu_tensor = CreateTensor<T>(shape, kDLCPU, 0);
  core::Tensor gpu_tensor = CreateTensor<T>(shape, kDLGPU, 0);
  RandomFillHost(cpu_tensor.mutableData<T>(), cpu_tensor.numel());
  core::Copy<T>(cpu_tensor, gpu_tensor);
  return std::make_tuple(std::move(cpu_tensor), std::move(gpu_tensor));
}

template <typename T>
bool CheckResultOfCPUAndGPU(const core::Tensor& cpu_tensor,
                            const core::Tensor& gpu_tensor) {
  TT_ENFORCE(layers::kernels::common::is_same_shape(cpu_tensor, gpu_tensor),
             "The shape of the inputs is not equal.");
  const T* cpu_data = cpu_tensor.data<T>();
  core::Tensor tmp_tensor = CreateTensor<T>({gpu_tensor.numel()}, kDLCPU, 0);
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
#endif

template <typename T>
bool CheckResultOfCPU(const core::Tensor& cpu_tensor_lhs,
                      const core::Tensor& cpu_tensor_rhs) {
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

}  // namespace common
}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
