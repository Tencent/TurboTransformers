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
  // TODO
}

void cblas_sgemm(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE* transa_array,
                 const CBLAS_TRANSPOSE* transb_array, const blasint M,
                 const blasint N, const blasint K_a, float alpha, float* A,
                 const blasint lda, float* B, const blasint ldb, float beta,
                 float* C, const blasint ldc) {}
}
