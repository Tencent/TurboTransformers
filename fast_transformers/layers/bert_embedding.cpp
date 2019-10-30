#include "fast_transformers/layers/bert_embedding.h"
#include "fast_transformers/core/math_function.h"
#include "fast_transformers/layers/kernels/layer_norm.h"
namespace fast_transformers {
namespace layers {

static void LookupEmbedding(const float *word_embeddings,
                            const int64_t *tokens_ids, float *out,
                            int64_t batch_size, int64_t seq_length,
                            int64_t vocab_size, int64_t hidden_size) {
  for (size_t i = 0; i < batch_size * seq_length; ++i) {
    int id = tokens_ids[i];
    FT_ENFORCE_LT(id, vocab_size, "embedding id out of index");
    float *dst = out + i * hidden_size;
    const float *src = word_embeddings + id * hidden_size;
    std::copy(src, src + hidden_size, dst);
  }
}

core::Tensor BERTEmbedding::
operator()(const core::Tensor &input_ids, const core::Tensor &position_ids,
           const core::Tensor &token_type_ids) const {
  if (VLOG_IS_ON(3)) {
    std::ostringstream os;
    os << ">>>>>>>>>>>> input_ids <<<<<<<<<<<<" << std::endl;
    input_ids.Print<int64_t>(os);
    os << ">>>>>>>>>>>> position_ids <<<<<<<<<<<<" << std::endl;
    position_ids.Print<int64_t>(os);
    os << ">>>>>>>>>>>> token_type_ids <<<<<<<<<<<<" << std::endl;
    token_type_ids.Print<int64_t>(os);
    VLOG(3) << os.str();
  }
  FT_ENFORCE_EQ(
      input_ids.n_dim(), 2,
      "The input ids should be a matrix with shape [BatchSize, SeqLen].");
  auto batch_size = input_ids.shape(0);
  auto seq_length = input_ids.shape(1);
  // TODO 1. switch DeviceType::CPU 2. how should I set stride?
  auto hidden_size = word_embedings_.shape(1);
  core::Tensor output_tensor(
      core::CreateDLPackTensor<float>({batch_size, seq_length, hidden_size}));

  auto vocab_size = word_embedings_.shape(0);

  LookupEmbedding(word_embedings_.data<float>(), input_ids.data<int64_t>(),
                  output_tensor.mutableData<float>(), batch_size, seq_length,
                  vocab_size, hidden_size);
  auto token_type_ids_ptr = token_type_ids.data<int64_t>();

  auto token_type_vocab_size = token_type_embeddings_.shape(0);
  std::vector<float> data(batch_size * seq_length * token_type_vocab_size, 0.);
  for (size_t i = 0; i < batch_size * seq_length; ++i) {
    auto id = token_type_ids_ptr[i];
    FT_ENFORCE_LT(id, token_type_vocab_size, "Out of range");
    data[i * token_type_vocab_size + id] = 1.;
  }
  core::cpu_blas_gemm(false, false, hidden_size, seq_length * batch_size,
                      token_type_vocab_size, 1.f,
                      token_type_embeddings_.data<float>(), hidden_size,
                      &data[0], token_type_vocab_size, 1.f,
                      output_tensor.mutableData<float>(), hidden_size);

  FT_ENFORCE_LT(seq_length, position_embeddings_.shape(0),
                "Seq length out of range");
  {
    auto *out = output_tensor.mutableData<float>();
    auto position_embeddings = position_embeddings_.data<float>();
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < seq_length; ++j) {
        for (size_t k = 0; k < hidden_size; ++k) {
          out[(i * seq_length + j) * hidden_size + k] +=
              position_embeddings[j * hidden_size + k];
        }
      }
    }
  }
  kernels::LayerNorm(
      output_tensor.mutableData<float>(), layer_norm_weights_.data<float>(),
      layer_norm_bias_.data<float>(), batch_size * seq_length, hidden_size);

  return output_tensor;
}
void BERTEmbedding::EnforceShapeAndType() const {

  VLOG(3) << ">>>>> init BERTEmbedding <<<<<<<<";
  FT_ENFORCE_EQ(word_embedings_.GetDeviceType(), kDLCPU, "Only CPU supportted");

  VLOG(3) << ">>>>> Assert OK <<<<<<<<";

  if (VLOG_IS_ON(3)) {
    std::ostringstream os;
    os << ">>>>> word_embedings_ <<<<<<<<" << std::endl;
    word_embedings_.Print<float>(os);
    os << ">>>>> position_embeddings_ <<<<<<<<" << std::endl;
    position_embeddings_.Print<float>(os);
    os << ">>>>> token_type_embeddings_ <<<<<<<<" << std::endl;
    token_type_embeddings_.Print<float>(os);
    os << ">>>>> layer_norm_weights_ <<<<<<<<" << std::endl;
    layer_norm_weights_.Print<float>(os);
    os << ">>>>> layer_norm_bias_ <<<<<<<<" << std::endl;
    layer_norm_bias_.Print<float>(os);
    VLOG(3) << os.str();
  }
}
} // namespace layers
} // namespace fast_transformers
