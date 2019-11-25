#pragma once
#include "fast_transformers/core/tensor.h"
namespace fast_transformers {
namespace layers {
namespace kernels {
extern void Matmul(const core::Tensor& A, bool a_trans, const core::Tensor& B,
                   bool b_trans, float alpha, core::Tensor* out, float beta);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
