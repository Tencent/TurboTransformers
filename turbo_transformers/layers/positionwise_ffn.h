

#pragma once
#include <memory>
#include <utility>

#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace layers {

class PositionwiseFeedForward {
 public:
  PositionwiseFeedForward(core::Tensor dense_weight_1,
                          core::Tensor dense_bias_1,
                          core::Tensor dense_weight_2,
                          core::Tensor dense_bias_2,
                          core::Tensor layer_norm_weight,
                          core::Tensor layer_norm_bias)
      : dense_weight_1_(std::move(dense_weight_1)),
        dense_bias_1_(std::move(dense_bias_1)),
        dense_weight_2_(std::move(dense_weight_2)),
        dense_bias_2_(std::move(dense_bias_2)),
        layer_norm_weight_(std::move(layer_norm_weight)),
        layer_norm_bias_(std::move(layer_norm_bias)) {
    EnforceShapeAndType();
  }
  void EnforceShapeAndType() const;

  // according to profiling results on Intel 61xx, is_trans_weight = true is
  // faster
  void operator()(const core::Tensor &input_tensor, core::Tensor *output,
                  bool is_trans_weight = true) const;

 private:
  core::Tensor dense_weight_1_;
  core::Tensor dense_bias_1_;
  core::Tensor dense_weight_2_;
  core::Tensor dense_bias_2_;
  core::Tensor layer_norm_weight_;
  core::Tensor layer_norm_bias_;
};

class DistrillFFN {
 public:
  DistrillFFN(core::Tensor dense_weight_1, core::Tensor dense_bias_1,
              core::Tensor dense_weight_2, core::Tensor dense_bias_2,
              core::Tensor layer_norm_weight, core::Tensor layer_norm_bias)
      : dense_weight_1_(std::move(dense_weight_1)),
        dense_bias_1_(std::move(dense_bias_1)),
        dense_weight_2_(std::move(dense_weight_2)),
        dense_bias_2_(std::move(dense_bias_2)),
        layer_norm_weight_(std::move(layer_norm_weight)),
        layer_norm_bias_(std::move(layer_norm_bias)) {
    EnforceShapeAndType();
  }
  void EnforceShapeAndType() const;

  // according to profiling results on Intel 61xx, is_trans_weight = true is
  // faster
  void operator()(const core::Tensor &input_tensor, core::Tensor *output,
                  bool is_trans_weight = true) const;

 private:
  core::Tensor dense_weight_1_;
  core::Tensor dense_bias_1_;
  core::Tensor dense_weight_2_;
  core::Tensor dense_bias_2_;
  core::Tensor layer_norm_weight_;
  core::Tensor layer_norm_bias_;
};

}  // namespace layers
}  // namespace turbo_transformers
