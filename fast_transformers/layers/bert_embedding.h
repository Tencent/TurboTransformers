#pragma once
#include "fast_transformers/core/tensor.h"
#include "fast_transformers/ops/cpu_bert_embedding_op.h"
#include <memory>

namespace fast_transformers {
namespace layers {

class BERTEmbedding {
public:
  BERTEmbedding(core::Tensor word_embeddings, core::Tensor position_embeddings,
                core::Tensor token_type_embeddings,
                core::Tensor layer_norm_weights, core::Tensor layer_norm_bias,
                float dropout_rate)
      : word_embedings_(
            std::move(word_embeddings)), // [vocab_size, hidden_size]
        position_embeddings_(std::move(
            position_embeddings)), // [max_position_embeddings, hidden_size]
        token_type_embeddings_(std::move(
            token_type_embeddings)), // [token_type_vocab_size, hidden_size]
        layer_norm_weights_(std::move(layer_norm_weights)), // [hidden_size]
        layer_norm_bias_(std::move(layer_norm_bias)),       // [hidden_size]
        dropout_rate_(dropout_rate) {
    std::cerr << ">>>>> init BERTEmbedding <<<<<<<<";
    assert(word_embedings_.GetDeviceType() == kDLCPU);
    assert(word_embedings_.GetDataTypeCode() == kDLFloat);

    std::cerr << ">>>>> Assert OK <<<<<<<<" << std::endl;

    vocab_size_ = word_embedings_.shape(0);
    hidden_size_ = word_embedings_.shape(1);
    token_type_vocab_size_ = token_type_embeddings_.shape(0);
    max_position_embeddings_ = position_embeddings_.shape(0);

    std::cerr << "vocab_size: " << vocab_size_ << ", "
              << "hidden_size: " << hidden_size_ << ", "
              << "token_type_vocab_size: " << token_type_vocab_size_ << ", "
              << "max_position_embeddings: " << max_position_embeddings_
              << std::endl;

    embedding_lookup_op_ =
        std::make_shared<ops::EmbeddingLookupOp<float, kDLCPU>>(vocab_size_,
                                                                hidden_size_);
    embedding_postprocessor_op_ =
        std::make_shared<ops::EmbeddingPostprocessorOp<float, kDLCPU>>(
            token_type_vocab_size_, hidden_size_, max_position_embeddings_);
    std::cerr << ">>>>> word_embedings_ <<<<<<<<" << std::endl;
    word_embedings_.Print<float>(std::cout);
    std::cerr << ">>>>> position_embeddings_ <<<<<<<<" << std::endl;
    position_embeddings_.Print<float>(std::cout);
    std::cerr << ">>>>> token_type_embeddings_ <<<<<<<<" << std::endl;
    token_type_embeddings_.Print<float>(std::cout);
    std::cerr << ">>>>> layer_norm_weights_ <<<<<<<<" << std::endl;
    layer_norm_weights_.Print<float>(std::cout);
    std::cerr << ">>>>> layer_norm_bias_ <<<<<<<<" << std::endl;
    layer_norm_bias_.Print<float>(std::cout);

    std::cerr << ">>>>> loading data <<<<<<<<" << std::endl;
    embedding_lookup_op_->initialize(word_embedings_.data<float>());
    embedding_postprocessor_op_->initialize(
        position_embeddings_.data<float>(),
        token_type_embeddings_.data<float>(),
        layer_norm_weights_.data<float>(), // TODO beta?
        layer_norm_bias_.data<float>());   // gemma?
    std::cerr << ">>>>> finish load data <<<<<<<<" << std::endl;
  }

  core::Tensor operator()(const core::Tensor &input_ids,
                          const core::Tensor &position_ids,
                          const core::Tensor &token_type_ids) const;

private:
  unsigned vocab_size_;
  unsigned hidden_size_;
  unsigned token_type_vocab_size_;
  unsigned max_position_embeddings_;

  core::Tensor word_embedings_;
  core::Tensor position_embeddings_;
  core::Tensor token_type_embeddings_;
  core::Tensor layer_norm_weights_;
  core::Tensor layer_norm_bias_;
  float dropout_rate_;

  // TODO should not known the data type and device type utill constructor is
  // executed. now hardcode for CPU and float, we should have an interface here.
  std::shared_ptr<ops::EmbeddingLookupOp<float, kDLCPU>> embedding_lookup_op_;
  std::shared_ptr<ops::EmbeddingPostprocessorOp<float, kDLCPU>>
      embedding_postprocessor_op_;
};

} // namespace layers
} // namespace fast_transformers
