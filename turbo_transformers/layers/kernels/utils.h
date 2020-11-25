

#pragma once
#include "turbo_transformers/core/tensor.h"
namespace turbo_transformers {
namespace layers {
namespace kernels {

void AddBias(const core::Tensor& bias, core::Tensor* output,
             const std::string name = "AddBias");
void AddInputBias(const core::Tensor& input1, const core::Tensor& input2,
                  const core::Tensor& bias, core::Tensor* output,
                  const std::string name = "AddInputBias");
template <typename T>
void Concat(const core::Tensor& t1, const core::Tensor& t2, size_t dim,
            core::Tensor* output, const std::string name = "Concat");

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
