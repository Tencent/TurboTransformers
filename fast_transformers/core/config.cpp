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

#include "fast_transformers/core/config.h"

namespace fast_transformers {
namespace core {
bool IsWithCUDA() {
#ifdef FT_WITH_CUDA
  return true;
#else
  return false;
#endif
}
BlasProvider GetBlasProvider() {
#ifdef FT_BLAS_USE_MKL
  return BlasProvider::MKL;
#elif defined(FT_BLAS_USE_OPENBLAS)
  return BlasProvider::OpenBlas;
#else
#error "unexpected code";
#endif
}
}  // namespace core
}  // namespace fast_transformers
