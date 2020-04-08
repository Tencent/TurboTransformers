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

#include "turbo_transformers/core/config.h"
#ifdef _OPENMP
#include "omp.h"
#endif
#include "blas.h"
#include "turbo_transformers/core/blas.h"

namespace turbo_transformers {
namespace core {
void SetNumThreads(int n_th) {
// The order seems important. Set MKL NUM_THREADS before OMP.
#ifdef TT_BLAS_USE_MKL
  mkl_set_num_threads(n_th);
#else
  openblas_set_num_threads(n_th);
#endif
#ifdef _OPENMP
  omp_set_num_threads(n_th);
#endif
}

BlasProvider GetBlasProvider() {
#ifdef TT_BLAS_USE_MKL
  return BlasProvider::MKL;
#elif defined(TT_BLAS_USE_OPENBLAS)
  return BlasProvider::OpenBlas;
#else
#error "unexpected code";
#endif
}

}  // namespace core
}  // namespace turbo_transformers
