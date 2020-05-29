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
#include "blas.h"
#define EIGEN_DONT_PARALLELIZE
#include "unsupported/Eigen/CXX11/Tensor"
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
                       const blasint* group_size) {
  int idx = 0;
  for (int i = 0; i < group_count; ++i) {
    auto alpha = alpha_array[i];
    auto beta = beta_array[i];
    for (int j = 0; j < group_size[i]; ++j) {
      cblas_sgemm(Layout, transa_array[i], transb_array[i], m_array[i],
                  n_array[i], k_array[i], alpha, a_array[idx], lda_array[i],
                  b_array[idx], ldb_array[i], beta, c_array[idx], ldc_array[i]);
      ++idx;
    }
  }
}

using Vec = Eigen::TensorMap<Eigen::Tensor<float, 1>>;

void vsTanh(blasint N, const float* in, float* out) {
  Vec input(const_cast<float*>(in), N);
  Vec output(out, N);

  // let use eigen to calculate tanh.
  // Eigen can use `FAST_MATH`.
  output = input.tanh();
}
}
