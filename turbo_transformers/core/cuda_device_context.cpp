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

#include "turbo_transformers/core/cuda_device_context.h"

#include "turbo_transformers/core/cuda_enforce.cuh"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/core/memory.h"

namespace turbo_transformers {
namespace core {

CUDADeviceContext::CUDADeviceContext() {
  TT_ENFORCE_CUDA_SUCCESS(cudaStreamCreate(&stream_));
  TT_ENFORCE_CUDA_SUCCESS(cublasCreate(&handle_));
  TT_ENFORCE_CUDA_SUCCESS(cublasSetStream(handle_, stream_));
  TT_ENFORCE_CUDA_SUCCESS(cudaGetDeviceProperties(&device_prop_, 0));
}

void CUDADeviceContext::Wait() const {
  cudaError_t e_sync = cudaSuccess;
  e_sync = cudaStreamSynchronize(stream_);
  TT_ENFORCE_CUDA_SUCCESS(e_sync);
}

cudaStream_t CUDADeviceContext::stream() const { return stream_; }

cublasHandle_t CUDADeviceContext::cublas_handle() const { return handle_; }

int CUDADeviceContext::compute_major() const { return device_prop_.major; }

CUDADeviceContext::~CUDADeviceContext() {
  Wait();
  TT_ENFORCE_CUDA_SUCCESS(cublasDestroy(handle_));
  TT_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(stream_));
}

}  // namespace core
}  // namespace turbo_transformers
