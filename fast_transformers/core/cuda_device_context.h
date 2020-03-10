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

#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory.h>

#include <map>

#include "macros.h"

namespace fast_transformers {
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

 private:
  cudaStream_t stream_;
  cublasHandle_t handle_;
  DISABLE_COPY_AND_ASSIGN(CUDADeviceContext);
};

}  // namespace core
}  // namespace fast_transformers
