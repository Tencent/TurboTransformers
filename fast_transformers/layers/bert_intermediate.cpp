#include "fast_transformers/layers/bert_intermediate.h"

#include <loguru.hpp>

#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/blas.h"
#include "fast_transformers/core/memory.h"
#include "fast_transformers/layers/kernels/activation.h"
#include "fast_transformers/layers/kernels/layer_norm.h"
#include "fast_transformers/layers/kernels/softmax.h"
#include "fast_transformers/layers/kernels/transpose.h"

namespace fast_transformers {
namespace layers {

namespace details {}  // namespace details

void BertIntermediate::operator()(const core::Tensor& input_tensor,
                                  core::Tensor* output_tensor) const {
  auto intermediate_size = dense_weight_.shape(0);  //[3072, 768]
  auto hidden_size = dense_weight_.shape(1);
  auto batch_size = input_tensor.shape(0);
  auto seq_length = input_tensor.shape(1);
  auto m = batch_size * seq_length;
  auto k = hidden_size;
  auto n = intermediate_size;
  static constexpr float alpha = 1., beta = 0.;

  FT_ENFORCE(output_tensor, "The output tensor should not be nullptr.");
  output_tensor->Reshape({batch_size, seq_length, intermediate_size});

  core::cblas_sgemm(core::CblasRowMajor, core::CblasNoTrans, core::CblasTrans,
                    m, n, k, alpha, input_tensor.data<float>(), k,
                    dense_weight_.data<float>(), k, beta,
                    output_tensor->mutableData<float>(), n);

  kernels::AddBiasGeLUAct(output_tensor->mutableData<float>(),
                          dense_bias_.data<float>(), m, n);
}

void BertIntermediate::EnforceShapeAndType() const {
  FT_ENFORCE_EQ(dense_weight_.device_type(), kDLCPU, "Only CPU supportted");
  FT_ENFORCE_EQ(dense_weight_.n_dim(), 2, "dense weight must be matrix");
  FT_ENFORCE_EQ(dense_bias_.n_dim(), 1, "dense bias must be vector");
  FT_ENFORCE_EQ(dense_weight_.shape(0), dense_bias_.shape(0),
                "weight and bias shape mismatch %d, %d", dense_weight_.shape(0),
                dense_bias_.shape(0));

  if (loguru::current_verbosity_cutoff() >= 3) {
    std::ostringstream os;
    os << ">>>>>>>>>>>> query_weight <<<<<<<<<<<<" << std::endl;
    dense_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> query_bias <<<<<<<<<<<<" << std::endl;
    dense_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}

}  // namespace layers
}  // namespace fast_transformers
