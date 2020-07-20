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

#if defined(TT_BLAS_USE_MKL)
#include "mkl.h"

namespace turbo_transformers {
using BlasInt = MKL_INT;
}

#elif defined(TT_BLAS_USE_OPENBLAS) || defined(TT_BLAS_USE_BLIS)
#include "cblas.h"
#if defined(TT_BLAS_USE_OPENBLAS)

namespace turbo_transformers {
using BlasInt = blasint;
}  // namespace turbo_transformers
#elif defined(TT_BLAS_USE_BLIS)
#include <unistd.h>

namespace turbo_transformers {
using BlasInt = f77_int;
}  // namespace turbo_transformers

using blasint = turbo_transformers::BlasInt;
#endif

extern "C" {
void cblas_sgemm_batch(const CBLAS_ORDER Layout,
                       const CBLAS_TRANSPOSE* transa_array,
                       const CBLAS_TRANSPOSE* transb_array,
                       const blasint* m_array, const blasint* n_array,
                       const blasint* k_array, const float* alpha_array,
                       const float** a_array, const blasint* lda_array,
                       const float** b_array, const blasint* ldb_array,
                       const float* beta_array, float** c_array,
                       const blasint* ldc_array, const blasint group_count,
                       const blasint* group_size);
void vsTanh(blasint N, const float* in, float* out);
}
#else
#endif
