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

#include "fast_transformers/core/cuda_device_context.h"

#include "fast_transformers/core/enforce.h"
#include "fast_transformers/core/memory.h"

namespace fast_transformers {
namespace core {

CUDADeviceContext::CUDADeviceContext() {
  cudaStreamCreate(&stream_);
  cublas_handle_.reset(new CublasHandleHolder(stream_));
}

void CUDADeviceContext::Wait() const {
  cudaError_t e_sync = cudaSuccess;
  e_sync = cudaStreamSynchronize(stream_);
  FT_ENFORCE_CUDA_SUCCESS(e_sync);
}

cudaStream_t CUDADeviceContext::stream() const { return stream_; }

CUDADeviceContext::~CUDADeviceContext() {
  Wait();
  cublas_handle_.reset();
  FT_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(stream_));
}

}  // namespace core
}  // namespace fast_transformers
