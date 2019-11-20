#pragma once
#if FT_BLAS_PROVIDER == MKL
#include "mkl.h"

namespace fast_transformers {
using BlasInt = MKL_INT;
}

#else
#error "Not implemented"
#endif
