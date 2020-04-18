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
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory.h>

#include <map>

#include "macros.h"

namespace turbo_transformers {
namespace core {

class CUDADeviceContext {
 public:
  CUDADeviceContext();

  ~CUDADeviceContext();

  static CUDADeviceContext& GetInstance() {
    static CUDADeviceContext instance;
    return instance;
  }

  void Wait() const;

  cudaStream_t stream() const;

  cublasHandle_t cublas_handle() const;

  int compute_major() const;

 private:
  cudaStream_t stream_;
  cublasHandle_t handle_;
  cudaDeviceProp device_prop_;
  DISABLE_COPY_AND_ASSIGN(CUDADeviceContext);
};

}  // namespace core
}  // namespace turbo_transformers
