

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
