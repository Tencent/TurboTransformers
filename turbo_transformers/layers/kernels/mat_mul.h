

#pragma once
#include "turbo_transformers/core/tensor.h"
namespace turbo_transformers {
namespace layers {
namespace kernels {
extern void MatMul(const core::Tensor& A, bool a_trans, const core::Tensor& B,
                   bool b_trans, float alpha, core::Tensor* out, float beta,
                   const std::string name = "MatMul");
extern void BatchMatMul(const core::Tensor& A, bool a_trans,
                        const core::Tensor& B, bool b_trans, float alpha,
                        core::Tensor* C, float beta,
                        const std::string name = "BatchMatMul");

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
