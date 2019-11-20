#include "fast_transformers/layers/bert_intermediate.h"

#include <loguru.hpp>

#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/blas.h"
#include "fast_transformers/core/eigen-tensor.h"
#include "fast_transformers/core/memory.h"
#include "fast_transformers/layers/kernels/activation.h"
#include "fast_transformers/layers/kernels/layer_norm.h"
#include "fast_transformers/layers/kernels/softmax.h"
#include "fast_transformers/layers/kernels/transpose.h"

namespace fast_transformers {
namespace layers {

void BertIntermediate::operator()(const core::Tensor& input_tensor,
                                  core::Tensor* output_tensor) const {
  output_tensor->Reshape<float>(
      {input_tensor.shape(0), input_tensor.shape(1), dense_weight_.shape(0)});

  auto X = core::to_mat(input_tensor);
  auto W = core::to_mat(dense_weight_);
  auto B = core::to_vector(dense_bias_);
  auto out_mat = core::to_mat(output_tensor);

  out_mat.noalias() = (X * W.transpose()).rowwise() + B.transpose();
  auto out_mat_array = out_mat.array();

  out_mat_array = out_mat_array * 0.5f *
                  (1.0f + (0.7978845608028654f *
                           (out_mat_array + 0.044715f * out_mat_array *
                                                out_mat_array * out_mat_array))
                              .tanh());
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
