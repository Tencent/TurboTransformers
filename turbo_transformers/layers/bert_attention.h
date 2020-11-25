

#pragma once
#include <memory>
#include <mutex>
#include <utility>

#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/layers/multi_headed_attention.h"

namespace turbo_transformers {
namespace layers {

class BertAttention : public MultiHeadedAttention {
 public:
  BertAttention(core::Tensor qkv_weight, core::Tensor qkv_bias,
                core::Tensor dense_weight, core::Tensor dense_bias,
                core::Tensor layer_norm_weight, core::Tensor layer_norm_bias,
                int64_t num_attention_heads)
      : MultiHeadedAttention(
            std::move(core::Tensor(nullptr)), std::move(core::Tensor(nullptr)),
            std::move(core::Tensor(nullptr)), std::move(core::Tensor(nullptr)),
            std::move(core::Tensor(nullptr)), std::move(core::Tensor(nullptr)),
            std::move(dense_weight), std::move(dense_bias),
            std::move(qkv_weight), std::move(qkv_bias),
            std::move(layer_norm_weight),  //(768)
            std::move(layer_norm_bias), num_attention_heads) {
    EnforceShapeAndType();
  }
  void EnforceShapeAndType() const;

  void operator()(const core::Tensor &input_tensor,
                  const core::Tensor &attention_mask, core::Tensor *output,
                  core::Tensor *attn = nullptr,
                  bool is_trans_weight = false) const;
};

}  // namespace layers
}  // namespace turbo_transformers
